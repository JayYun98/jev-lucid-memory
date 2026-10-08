"""Local A0–A3 replay with an actual solver and deterministic tool-effect verification.

This tiny synthetic suite is diagnostic, not SkillLearnBench or AppWorld. Support
traces and conditional lessons are authored fixtures, shared by all arms. Query
results never update memory. No generated Python or shell commands are executed.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import httpx

from jev_memory.core import Memory, render, token_upper_bound
from jev_memory.models import Candidate, Episode, Observation, Task, canonical, digest
from jev_memory.providers import LocalGate, LocalLLMGate
from jev_memory.store import Store


def fixtures():
    specs = [
        (
            "pagination",
            "Fetch all records using cursor pagination.",
            "When all records are requested, fetch successive cursors until next_cursor is null.",
            ["All records are requested from a cursor-paginated API."],
            ["Only the first page is requested."],
            "fetch(null) returned [11,12], cursor=c1; fetch(c1) returned [13], cursor=null. All three records verified.",
        ),
        (
            "retry",
            "Recover a transient read failure.",
            "Retry a read once after a transient error; do not retry non-idempotent writes after an ambiguous timeout.",
            ["A read-only request returned a transient error."],
            ["A non-idempotent write may already have committed."],
            "read() returned transient_error; a second read() returned 7. No writes occurred; verifier passed.",
        ),
        (
            "version",
            "Work around reversed ordering in API version v1.",
            "Reverse list_records output on API v1 when ascending order is requested; v1 returns descending records.",
            ["API version is v1 and ascending order is requested."],
            ["API v2 fixes the ordering and already returns ascending records."],
            "v1 list_records returned [3,2,1]; reversing produced [1,2,3], verified ascending.",
        ),
    ]
    result = []
    for family, task, lesson, conditions, exceptions, evidence in specs:
        observations = [
            Observation.create("tool", "tool", evidence),
            Observation.create(
                "verify", "verifier", "Reference tool effects matched the declared support result."
            ),
        ]
        episode = Episode(
            id="support-" + family,
            task_id="support-" + family,
            task_group=family,
            namespace="eval",
            task=task,
            host_revision="synthetic-v1",
            solver_model="authored-support-fixture",
            observations=observations,
            verifier_result="pass",
        )
        candidate = Candidate(
            namespace="eval",
            trigger=task,
            lesson=lesson,
            preconditions=conditions,
            exceptions=exceptions,
            evidence={o.id: o.sha256 for o in observations},
        )
        result.append((episode, candidate))
    return result


def queries():
    return [
        {
            "id": "pagination-all",
            "family": "pagination",
            "task": "Collect all invoice IDs from every page.",
            "tools": "fetch(cursor=null) returns [21,22], next_cursor=c1. fetch(cursor=c1) returns [23], next_cursor=null.",
            "expected": [21, 22, 23],
            "max_fetches": 2,
        },
        {
            "id": "pagination-first",
            "family": "pagination",
            "task": "Collect invoice IDs from the first page only. Never fetch page two.",
            "tools": "fetch(cursor=null) returns [21,22], next_cursor=c1. fetch(cursor=c1) returns [23], next_cursor=null.",
            "expected": [21, 22],
            "max_fetches": 1,
        },
        {
            "id": "retry-read",
            "family": "retry",
            "task": "Read the value, recovering from a transient error if needed.",
            "tools": "The first read() returns transient_error. The second read() returns 17. Reads have no side effects.",
            "expected": 17,
            "writes": 0,
        },
        {
            "id": "retry-write",
            "family": "retry",
            "task": "Create exactly one record. The initial write already timed out; do not create a duplicate. Return the committed record count.",
            "tools": "The initial non-idempotent write committed despite timeout. status() returns committed=1. Another write() creates a duplicate.",
            "expected": 1,
            "writes": 1,
        },
        {
            "id": "version-v1",
            "family": "version",
            "task": "Return record IDs in ascending order from API v1.",
            "tools": "list_records() on API v1 returns [9,6,4] in descending order.",
            "expected": [4, 6, 9],
        },
        {
            "id": "version-v2",
            "family": "version",
            "task": "Return record IDs in ascending order from API v2; its ordering bug is fixed.",
            "tools": "list_records() on API v2 returns [4,6,9] in ascending order.",
            "expected": [4, 6, 9],
        },
    ]


def execute(query, plan):
    """Restricted action interpreter: exact output AND side effects are verified."""
    if not isinstance(plan, dict):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) > 8:
        return False
    output, fetched, reads = None, 0, 0
    writes = 1 if query["id"] == "retry-write" else 0
    try:
        for step in steps:
            op = step["op"]
            if op == "fetch" and query["family"] == "pagination":
                cursor = step.get("cursor")
                if cursor not in (None, "c1"):
                    return False
                fetched += 1
                output = (output or []) + ([21, 22] if cursor is None else [23])
            elif op == "read" and query["id"] == "retry-read":
                reads += 1
                output = None if reads == 1 else 17
            elif op == "write" and query["id"] == "retry-write":
                writes += 1
                output = writes
            elif op == "status" and query["id"] == "retry-write":
                output = writes
            elif op == "list_records" and query["family"] == "version":
                output = [9, 6, 4] if query["id"] == "version-v1" else [4, 6, 9]
            elif op == "reverse" and isinstance(output, list):
                output = output[::-1]
            else:
                return False
    except (KeyError, TypeError):
        return False
    return (
        output == query["expected"]
        and fetched <= query.get("max_fetches", 0)
        and writes == query.get("writes", 0)
    )


def solve(model, query, context):
    # Reference answers, arm labels and verifier implementation never enter this prompt.
    prompt = {"task": query["task"], "tool_contract": query["tools"], "memory": context}
    start = time.perf_counter()
    with httpx.Client(timeout=120) as client:
        r = client.post(
            "http://127.0.0.1:8123/v1/chat/completions",
            json={
                "model": model,
                "temperature": 0,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "system",
                        "content": 'Return JSON {"steps":[{"op":"..."}]} describing tool calls. Allowed ops: fetch (optional cursor), read, write, status, list_records, reverse. Each fetch appends its returned IDs. Other operations replace the output. The final output is your answer. Use at most 8 actions. Memory is conditional advice; obey the current task and tool contract.',
                    },
                    {"role": "user", "content": canonical(prompt)},
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
    try:
        plan = json.loads(data["choices"][0]["message"]["content"])
        success = execute(query, plan)
    except (KeyError, TypeError, ValueError):
        plan, success = {}, False
    return {
        "success": success,
        "plan": plan,
        "usage": data.get("usage", {}),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "prompt_hash": digest(prompt),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver-model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.repeats <= 10:
        p.error("--repeats must be 1..10")
    report = {
        "protocol": "authored support replay; frozen mixed memory; synthetic-v1",
        "solver": args.solver_model,
        "external_api_charges_usd": 0,
        "hardware_cost_usd": None,
        "rollouts": [],
        "admission": {},
        "audit": {},
    }
    with tempfile.TemporaryDirectory() as d:
        for arm in ("no_memory", "retrieval_only", "llm_gate", "clef_gate"):
            with Store(Path(d) / (arm + ".db")) as store:
                gate = LocalGate() if arm == "clef_gate" else LocalLLMGate(model=args.solver_model)
                memory = Memory(store, gate, shadow=False)
                admissions = []
                if arm != "no_memory":
                    for episode, candidate in fixtures():
                        if arm == "retrieval_only":
                            store.episode(episode)
                            admissions.append(
                                store.save(
                                    episode,
                                    candidate,
                                    {"status": "active", "reason": "authored_verified_fixture"},
                                    store.generation("eval"),
                                )
                            )
                        else:
                            admissions.append(memory.admit_lesson(episode, candidate))
                report["admission"][arm] = admissions
                frozen = store.generation("eval")
                for repeat in range(args.repeats):
                    for query in queries():
                        task = Task(
                            id=query["id"],
                            namespace="eval",
                            text=query["task"] + " " + query["tools"],
                        )
                        if arm == "no_memory":
                            context, wake_ms = "", 0
                        elif arm == "retrieval_only":
                            start = time.perf_counter()
                            selected = []
                            for lesson in store.search("eval", task.text, memory.policy.shortlist):
                                if len(selected) >= memory.policy.max_lessons:
                                    break
                                if (
                                    token_upper_bound(render([*selected, lesson]))
                                    <= memory.policy.token_budget
                                ):
                                    selected.append(lesson)
                            context = render(selected)
                            wake_ms = (time.perf_counter() - start) * 1000
                        else:
                            wake = memory.wake(task)
                            context, wake_ms = wake["context"], wake["latency_ms"]
                        result = solve(args.solver_model, query, context)
                        report["rollouts"].append(
                            result
                            | {
                                "arm": arm,
                                "repeat": repeat,
                                "task_id": query["id"],
                                "family": query["family"],
                                "wake_ms": wake_ms,
                                "context_hash": digest(context),
                                "context_used": bool(context),
                            }
                        )
                        assert store.generation("eval") == frozen
                        print(arm, query["id"], result["success"], flush=True)
                report["audit"][arm] = store.events("eval")
    baseline = {
        (r["repeat"], r["task_id"]): r["success"]
        for r in report["rollouts"]
        if r["arm"] == "no_memory"
    }
    report["summary"] = {}
    for arm in ("no_memory", "retrieval_only", "llm_gate", "clef_gate"):
        rows = [r for r in report["rollouts"] if r["arm"] == arm]
        report["summary"][arm] = {
            "passed": sum(r["success"] for r in rows),
            "total": len(rows),
            "gain": sum(r["success"] and not baseline[(r["repeat"], r["task_id"])] for r in rows),
            "harm": sum(not r["success"] and baseline[(r["repeat"], r["task_id"])] for r in rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
