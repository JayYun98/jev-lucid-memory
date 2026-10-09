"""JSON in, JSON out. Shadow records local traces; active memory changes require --apply."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .batch_writer import BatchWriter
from .consolidation import ConsolidationPolicy, Consolidator, GradedEpisode
from .core import Memory
from .models import Candidate, Episode, Identifier, Policy, Task, canonical
from .providers import (
    LLMGate,
    LocalGate,
    LocalLLMGate,
    LocalWriter,
    OpenRouterGate,
    OpenRouterWriter,
    TypeSafeGate,
)
from .store import Conflict, Store


def parser():
    p = argparse.ArgumentParser(prog="jev-memory", description=__doc__)
    p.add_argument("--db", default="memory.db")
    p.add_argument(
        "--policy",
        type=Path,
        help="Optional validated JSON policy (calibrate on development tasks)",
    )
    p.add_argument(
        "--provider",
        choices=["openrouter", "typesafe", "llm", "local", "local-llm"],
        default="local",
    )
    p.add_argument("--endpoint", help="Loopback endpoint for local providers")
    p.add_argument("--model", help="Explicit decision model override")
    p.add_argument(
        "--apply", action="store_true", help="Activate admitted lessons / return Wake context"
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("sleep", "admit", "wake"):
        cmd = sub.add_parser(name)
        cmd.add_argument("file", type=Path, help="JSONL: one record per line")
        if name == "sleep":
            cmd.add_argument("--no-trigger", action="store_true")
            cmd.add_argument("--writer-model", default="local")
            cmd.add_argument(
                "--writer-endpoint", default="http://127.0.0.1:8123/v1/chat/completions"
            )
            cmd.add_argument(
                "--cloud-writer",
                action="store_true",
                help="Explicitly send episodes to OpenRouter for reflection",
            )
    for name in ("list", "audit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--namespace", required=True)
    cmd = sub.add_parser("batch-sleep", help="Review a JSONL batch of graded episodes")
    cmd.add_argument("file", type=Path)
    cmd.add_argument("--cycle-id", required=True)
    _batch_writer_args(cmd)
    cmd.add_argument("--consolidation-policy", type=Path)
    cmd = sub.add_parser("queue", help="List consolidation queue entries")
    cmd.add_argument("--namespace", required=True)
    cmd = sub.add_parser("retry", help="Retry a queued consolidation cycle")
    cmd.add_argument("id", type=int)
    cmd.add_argument("--namespace", required=True)
    _batch_writer_args(cmd)
    cmd.add_argument("--consolidation-policy", type=Path)
    cmd = sub.add_parser("archive")
    cmd.add_argument("--namespace", required=True)
    cmd.add_argument("id")
    cmd.add_argument("--revision", required=True, type=int)
    return p


def _batch_writer_args(cmd):
    cmd.add_argument("--writer-model", default="local")
    cmd.add_argument("--writer-endpoint", default="http://127.0.0.1:8123/v1/chat/completions")
    cmd.add_argument("--cloud-writer", action="store_true")


def _gate(args, p):
    cls = {
        "openrouter": OpenRouterGate,
        "typesafe": TypeSafeGate,
        "llm": LLMGate,
        "local": LocalGate,
        "local-llm": LocalLLMGate,
    }[args.provider]
    options = {"model": args.model} if args.model else {}
    if args.endpoint:
        if not args.provider.startswith("local"):
            p.error("--endpoint is only for local providers")
        options["endpoint"] = args.endpoint
    return cls(**options)


def _batch_result_code(result):
    return (
        1 if result.get("reason") in {"provider_error", "reviewer_error", "invalid_proposal"} else 0
    )


def _batch_command(args, p, store):
    if args.command == "queue":
        TypeAdapter(Identifier).validate_python(args.namespace)
        print(canonical(Consolidator(store, None).queue(args.namespace)))
        return 0
    policy = (
        ConsolidationPolicy.model_validate_json(args.consolidation_policy.read_text())
        if args.consolidation_policy
        else ConsolidationPolicy()
    )
    if args.command == "batch-sleep":
        # Parse the entire file before constructing providers or calling the model.
        TypeAdapter(Identifier).validate_python(args.cycle_id)
        episodes = [
            GradedEpisode.model_validate(json.loads(line))
            for line in args.file.read_text().splitlines()
            if line.strip()
        ]
        if (
            not episodes
            or len({e.episode.id for e in episodes}) != len(episodes)
            or any(e.episode.namespace != episodes[0].episode.namespace for e in episodes)
        ):
            raise ValueError("empty, duplicate, or mixed-namespace batch")
    else:
        TypeAdapter(Identifier).validate_python(args.namespace)
        if args.id < 1:
            raise ValueError("invalid queue ID")
        row = (
            store.db.execute(
                "SELECT namespace,status FROM consolidation_queue WHERE id=?", (args.id,)
            ).fetchone()
            if _has_queue(store)
            else None
        )
        if row is None or row["namespace"] != args.namespace or row["status"] == "resolved":
            raise ValueError("queue entry unavailable")
    gate = _gate(args, p)
    writer = (
        OpenRouterWriter(model=args.writer_model)
        if args.cloud_writer
        else LocalWriter(endpoint=args.writer_endpoint, model=args.writer_model)
    )
    consolidator = Consolidator(store, gate, policy, shadow=not args.apply)
    reviewer = BatchWriter(writer)
    result = (
        consolidator.run(episodes, reviewer, args.cycle_id)
        if args.command == "batch-sleep"
        else consolidator.retry(args.id, reviewer)
    )
    print(canonical(result))
    return _batch_result_code(result)


def _has_queue(store):
    return (
        store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='consolidation_queue'"
        ).fetchone()
        is not None
    )


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        with Store(args.db) as store:
            if args.command in {"batch-sleep", "queue", "retry"}:
                return _batch_command(args, p, store)
            if args.command in {"list", "audit", "archive"}:
                if args.command == "archive":
                    if not args.apply:
                        p.error("archive requires --apply")
                    store.transition(args.namespace, args.id, args.revision, "archived")
                    print(canonical({"status": "archived"}))
                    return 0
                print(
                    canonical(
                        store.lessons(args.namespace, active=False)
                        if args.command == "list"
                        else store.events(args.namespace)
                    )
                )
                return 0
            gate = _gate(args, p)
            policy = (
                Policy.model_validate_json(args.policy.read_text()) if args.policy else Policy()
            )
            memory = Memory(store, gate, policy, shadow=not args.apply)
            # Validate the whole input before any model calls or active writes.
            records = []
            for line in args.file.read_text().splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                if args.command == "admit":
                    if set(raw) != {"episode", "candidate"}:
                        raise ValueError("admit expects episode and candidate")
                    records.append(
                        (
                            Episode.model_validate(raw["episode"]),
                            Candidate.model_validate(raw["candidate"]),
                        )
                    )
                else:
                    records.append(
                        (Task if args.command == "wake" else Episode).model_validate(raw)
                    )
            if args.command == "sleep":
                writer = (
                    OpenRouterWriter(model=args.writer_model)
                    if args.cloud_writer
                    else LocalWriter(endpoint=args.writer_endpoint, model=args.writer_model)
                )
            failed = False
            for record in records:
                if args.command == "wake":
                    result = memory.wake(record)
                elif args.command == "admit":
                    result = memory.admit_lesson(*record)
                else:
                    result = memory.sleep(record, writer, trigger=not args.no_trigger)
                failed = (
                    failed
                    or bool(result.get("error"))
                    or result.get("reason") == "provider_error"
                    or any(r.get("reason") == "provider_error" for r in result.get("results", []))
                )
                print(canonical(result))
            return 1 if failed else 0
    except (ValidationError, ValueError, Conflict, OSError) as exc:
        # ValidationError may contain the entire input, including sensitive traces.
        print(
            f"jev-memory: {type(exc).__name__}; check input schema, credentials and file access",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
