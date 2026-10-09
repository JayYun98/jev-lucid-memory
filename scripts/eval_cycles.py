"""Real local multi-cycle smoke study. Synthetic tasks, not a LAB reproduction.

No authored lessons: a local model solves support tasks; independent execution
checks and isolated local graders produce feedback; a local writer proposes memory.
The decision backend is explicitly local Clef or a local generative baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from jev_memory.batch_writer import BatchWriter
from jev_memory.consolidation import ConsolidationPolicy, Consolidator
from jev_memory.core import Memory
from jev_memory.cycles import (
    Case,
    Cycle,
    GradeResult,
    Protocol,
    SolveResult,
    VerifyResult,
    run_cycles,
)
from jev_memory.models import Observation, Policy, Task, canonical, digest
from jev_memory.providers import LocalGate, LocalLLMGate, LocalWriter, local_url
from jev_memory.store import Store

SOLVE_SYSTEM = (
    "You are a tool action planner. Read the task in the user JSON. Return one valid JSON object with exactly a steps array. "
    "Each step has op and optional cursor. Allowed operations: fetch with cursor null or c1; sort_records with no cursor. "
    "To export all pages, fetch null then c1. To return only the first page, fetch null only. "
    "Do not include prose, markdown, expected outputs, or repeat these instructions."
)


def grade_system(style):
    return (
        "You are an independent tool action grader. Read the task and answer in the user JSON. "
        "Return one valid JSON object with exactly two boolean fields: output and scope. "
        + (
            "Independently simulate the tools. "
            if style == 0
            else "Audit completeness and forbidden extra work. "
        )
        + "output is true only if actions return all requested records. scope is true only if actions obey the requested page limit and tool rules. "
        "Do not include prose, markdown, or repeat these instructions."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:8123/v1/chat/completions")
    p.add_argument("--decision", choices=["clef", "llm"], default="clef")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", default="cycles-smoke")
    args = p.parse_args()
    if not 1 <= args.cycles <= 10:
        p.error("--cycles must be 1..10")
    endpoint = local_url(args.endpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    specs = {}
    calls = []

    def chat(system, data, max_tokens=400):
        started = time.perf_counter()
        response = httpx.post(
            endpoint,
            timeout=180,
            json={
                "model": args.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": canonical(data)},
                ],
            },
        )
        response.raise_for_status()
        body = response.json()
        elapsed = (time.perf_counter() - started) * 1000
        calls.append({"usage": body.get("usage", {}), "latency_ms": elapsed})
        try:
            content = body["choices"][0]["message"]["content"]
            fenced = re.fullmatch(
                r"\s*```json\s*\n(.*?)\n```\s*", content, re.DOTALL | re.IGNORECASE
            )
            if fenced:
                content = fenced.group(1)
                calls[-1]["fenced_json"] = True
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            (args.output / f"chat-error-{len(calls)}.json").write_text(response.text)
            raise
        return parsed, body.get("usage", {}), elapsed

    def case(cycle, split, slot, mode):
        ident = f"c{cycle}-{split}-{slot}"
        numbers = [
            cycle * 100 + slot * 10 + 3,
            cycle * 100 + slot * 10 + 1,
            cycle * 100 + slot * 10 + 2,
        ]
        if mode == "sort":
            spec = {"mode": mode, "records": numbers}
            text = "Return these record IDs in ascending order using sort_records: " + str(numbers)
            tools = ["sort_records"]
        else:
            spec = {"mode": mode, "pages": [numbers[:2], numbers[2:]]}
            text = (
                "Export all records from every page."
                if mode == "all"
                else "Return only the first page; do not fetch another page."
            )
            text += f" fetch(null) returns {numbers[:2]}, next_cursor=c1; fetch(c1) returns {numbers[2:]}, next_cursor=null."
            tools = ["fetch"]
        specs[ident] = spec
        return Case(
            Task(id=ident, namespace=args.run_id + "/memory", text=text, tools=tools),
            "matter-" + ident,
            "new" if mode == "sort" else "familiar",
            ("output", "scope"),
        )

    cycles = []
    for i in range(args.cycles):
        support = tuple(case(i, "support", j, "all" if i % 2 == 0 else "first") for j in range(2))
        query = tuple(case(i, "query", j, mode) for j, mode in enumerate(["all", "first", "sort"]))
        cycles.append(Cycle(f"cycle-{i + 1}", support, query))

    def execute(task, answer):
        spec = specs[task.id]
        output, count = [], 0
        trace = []
        try:
            steps = answer["steps"]
            if not isinstance(steps, list) or len(steps) > 4:
                raise ValueError()
            for step in steps:
                if step["op"] == "fetch" and spec["mode"] != "sort":
                    cursor = step.get("cursor")
                    if cursor not in (None, "c1"):
                        raise ValueError()
                    records = spec["pages"][0 if cursor is None else 1]
                    next_cursor = "c1" if cursor is None else None
                    output += records
                    count += 1
                    trace.append(
                        f"fetch({cursor}) returned {records}, next_cursor="
                        f"{'null' if next_cursor is None else next_cursor}"
                    )
                elif step["op"] == "sort_records" and spec["mode"] == "sort":
                    output = sorted(spec["records"])
                    trace.append(f"sort_records returned {output}")
                else:
                    raise ValueError()
            expected = (
                sorted(spec["records"])
                if spec["mode"] == "sort"
                else spec["pages"][0] + spec["pages"][1]
                if spec["mode"] == "all"
                else spec["pages"][0]
            )
            return {
                "output": output == expected,
                "scope": count <= (1 if spec["mode"] == "first" else 2),
            }, trace
        except (KeyError, TypeError, ValueError):
            return {"output": False, "scope": False}, trace + ["invalid action plan"]

    def solve(task, context, settings):
        answer, usage, latency = chat(
            SOLVE_SYSTEM,
            {"task": task.text, "tools": task.tools, "memory": context},
        )
        criteria, trace = execute(task, answer)
        observation = Observation.create(task.id + "-tools", "tool", "; ".join(trace))
        print("solved", task.id, "memory", bool(context), criteria, flush=True)
        return SolveResult(
            answer,
            (observation,),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            latency,
            len(answer.get("steps", [])),
        )

    def verify(task, answer):
        criteria, trace = execute(task, answer)
        return VerifyResult(
            all(criteria.values()), canonical({"criteria": criteria, "trace": trace})
        )

    def grader(style):
        def grade(task, answer, ids):
            # Isolated requests receive no solver reasoning, memory, or other grader verdict.
            result, grade_usage, _ = chat(
                grade_system(style),
                {"task": task.text, "answer": answer},
            )
            if set(result) != set(ids) or any(type(x) is not bool for x in result.values()):
                raise ValueError("invalid grader response")
            grade.last_usage = grade_usage
            return GradeResult(result, ids[0])

        return grade

    gate = (
        LocalGate()
        if args.decision == "clef"
        else LocalLLMGate(endpoint=endpoint, model=args.model)
    )
    reviewer = BatchWriter(LocalWriter(endpoint=endpoint, model=args.model, timeout=180))
    consolidation_policy = ConsolidationPolicy(comparison_batch=4)
    with Store(args.output / "memory.db") as store:
        consolidator = Consolidator(store, gate, consolidation_policy, shadow=False)

        def sleep(episodes, _store, namespace):
            cycle_id = episodes[0].episode.host_revision
            result = consolidator.run(list(episodes), reviewer, cycle_id)
            print("sleep", cycle_id, result, flush=True)
            sleep.last_usage = reviewer.last_usage
            return {
                "result": result,
                "writer_usage": reviewer.last_usage.model_dump() if reviewer.last_usage else None,
                "writer_latency_ms": reviewer.last_latency_ms,
            }

        memory = Memory(store, gate, Policy(token_budget=12000, max_lessons=10), shadow=False)

        def select(task, bank):
            before = len(store.events(task.namespace))
            selected = [r["id"] for r in bank if memory.use_lesson(task, r)]
            events = store.events(task.namespace)[before:]
            amounts = [e["body"].get("usage", {}) for e in events if e.get("kind") == "retrieval"]
            select.last_usage = {
                "input_tokens": sum(u.get("input_tokens", 0) for u in amounts),
                "output_tokens": sum(u.get("output_tokens", 0) for u in amounts),
            }
            return selected

        protocol = Protocol(
            tuple(cycles),
            {
                "model": args.model,
                "temperature": 0,
                "max_tokens": 400,
                "solve_prompt_hash": digest(SOLVE_SYSTEM),
                "grader_prompt_hashes": [digest(grade_system(i)) for i in range(2)],
                "decision_backend": args.decision,
                "consolidation_policy": consolidation_policy.model_dump(mode="json"),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            16000,
            args.run_id,
            ("local-grader-a", "local-grader-b"),
        )
        result = run_cycles(
            protocol,
            manifest_path=args.output / "manifest.json",
            store=store,
            solve=solve,
            verify=verify,
            batch_sleep=sleep,
            select=select,
            graders={key: grader(i) for i, key in enumerate(protocol.grader_ids)},
        )
        sleep_failures = [
            {"cycle_id": cycle_id, "reason": cycle["sleep_cost"]["result"]["reason"]}
            for cycle_id, cycle in result["cycles"].items()
            if cycle.get("sleep_cost", {}).get("result", {}).get("reason")
            in {"reviewer_error", "provider_error", "invalid_proposal"}
        ]
        receipt = {
            "protocol": "local synthetic cycles; not LAB; same local model in isolated grader requests",
            "decision": args.decision,
            "external_api_charges_usd": 0,
            "hardware_energy_cost_usd": None,
            "chat_calls": calls,
            "events": store.events(args.run_id + "/memory"),
            "complete": result["complete"],
            "sleep_failures": sleep_failures,
        }
        (args.output / "receipt.json").write_text(canonical(receipt) + "\n")
    print("complete", args.output, flush=True)
    return 1 if sleep_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
