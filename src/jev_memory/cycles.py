"""Resumable, split-isolated multi-cycle evaluation over one admitted memory bank.

Adapters are callables so a local solver, deterministic verifier, batch Sleep and
independent graders can be supplied without coupling the protocol to a provider.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from .consolidation import GradedEpisode, RubricVerdict
from .core import eligible, render, token_upper_bound
from .models import Episode, Observation, Task, canonical, digest
from .store import Store

Arm = Literal["no_memory", "full_memory", "selected_memory"]
ARMS: tuple[Arm, ...] = ("no_memory", "full_memory", "selected_memory")


@dataclass(frozen=True)
class Case:
    task: Task
    matter_id: str
    cohort: Literal["familiar", "new"]
    rubric_ids: tuple[str, ...] = ()
    reference: str | None = None


@dataclass(frozen=True)
class Cycle:
    id: str
    support: tuple[Case, ...]
    query: tuple[Case, ...]


@dataclass(frozen=True)
class Protocol:
    cycles: tuple[Cycle, ...]
    solver_settings: Mapping[str, object]
    context_budget: int
    run_id: str
    grader_ids: tuple[str, str]
    disagreement: Literal["fail_closed", "exclude"] = "fail_closed"


@dataclass(frozen=True)
class SolveResult:
    answer: object
    observations: tuple[Observation, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    tool_calls: int = 0
    compute: float | None = None
    deliverables: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    details: str


@dataclass(frozen=True)
class GradeResult:
    criteria: Mapping[str, bool]
    rubric_id: str


def validate_protocol(protocol: Protocol) -> None:
    if not protocol.run_id or not protocol.cycles or protocol.context_budget <= 0:
        raise ValueError("run_id, cycles, and positive context_budget required")
    if len(set(protocol.grader_ids)) != 2:
        raise ValueError("two distinct graders required")
    cycle_ids: set[str] = set()
    seen_task: dict[str, str] = {}
    seen_matter: dict[str, str] = {}
    for cycle in protocol.cycles:
        if not cycle.id or cycle.id in cycle_ids or not cycle.support or not cycle.query:
            raise ValueError("unique cycle ID and nonempty support/query required")
        cycle_ids.add(cycle.id)
        for split, cases in (("support", cycle.support), ("query", cycle.query)):
            for case in cases:
                if not case.matter_id:
                    raise ValueError("matter_id required")
                if not case.rubric_ids:
                    raise ValueError("rubric_ids required")
                # A case, task or matter cannot cross phases/cycles. This intentionally
                # rejects repeated matter variants even when their text differs.
                for key, value, seen in (
                    ("task", case.task.id, seen_task),
                    ("matter", case.matter_id, seen_matter),
                ):
                    if value in seen:
                        raise ValueError(f"{key} leakage: {value}")
                    seen[value] = f"{cycle.id}/{split}"


def _case_data(case: Case) -> dict:
    return {
        "task": case.task.model_dump(mode="json"),
        "matter_id": case.matter_id,
        "cohort": case.cohort,
        "rubric_ids": list(case.rubric_ids),
        "reference": case.reference,
    }


def _protocol_data(protocol: Protocol) -> dict:
    return {
        "cycles": [
            {
                "id": c.id,
                "support": [_case_data(x) for x in c.support],
                "query": [_case_data(x) for x in c.query],
            }
            for c in protocol.cycles
        ],
        "solver_settings": dict(protocol.solver_settings),
        "context_budget": protocol.context_budget,
        "run_id": protocol.run_id,
        "grader_ids": list(protocol.grader_ids),
        "disagreement": protocol.disagreement,
    }


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as f:
        temp = Path(f.name)
        f.write(canonical(data) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _answer_data(value: object) -> object:
    # Serialization check before committing a completed attempt.
    canonical(value)
    return value


def _metric(result: SolveResult) -> dict:
    return {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "tool_calls": result.tool_calls,
        "compute": result.compute,
    }


def _bounded(lessons: Sequence[dict], budget: int) -> list[dict]:
    kept: list[dict] = []
    for lesson in lessons:
        if token_upper_bound(render([*kept, lesson])) <= budget:
            kept.append(lesson)
    return kept


def _usage(adapter) -> dict:
    value = getattr(adapter, "last_usage", None)
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("adapter last_usage must be a mapping or model")


def _grades(
    case: Case,
    answer: object,
    graders: Mapping[str, Callable],
    grader_ids: tuple[str, str],
    disagreement: str,
    verifier_passed: bool,
) -> dict:
    grades = {}
    grade_cost = {}
    for grader_id in grader_ids:
        started = time.perf_counter()
        grader = graders[grader_id]
        grade = grader(
            Task.model_validate(case.task.model_dump(mode="json")),
            deepcopy(answer),
            case.rubric_ids,
        )
        grade_cost[grader_id] = {
            "latency_ms": (time.perf_counter() - started) * 1000,
        } | _usage(grader)
        if grade.rubric_id not in case.rubric_ids or set(grade.criteria) != set(case.rubric_ids):
            raise ValueError("grader rubric mismatch")
        if any(type(value) is not bool for value in grade.criteria.values()):
            raise ValueError("grader criteria must be booleans")
        grades[grader_id] = dict(grade.criteria)
    left, right = (grades[grader_ids[0]], grades[grader_ids[1]])
    agreement = {rubric: left[rubric] == right[rubric] for rubric in case.rubric_ids}
    criteria = {
        rubric: (
            left[rubric]
            if agreement[rubric]
            else (False if disagreement == "fail_closed" else None)
        )
        for rubric in case.rubric_ids
    }
    all_pass = (
        False
        if not verifier_passed
        else None
        if any(value is None for value in criteria.values())
        else all(criteria.values())
    )
    return {
        "grades": grades,
        "grade_cost": grade_cost,
        "grader_agreement": all(agreement.values()),
        "criterion_agreement": agreement,
        "criterion_pass": criteria,
        "all_pass": all_pass,
    }


def _select(
    case: Case, bank: tuple[dict, ...], namespace: str, budget: int, select: Callable
) -> tuple[list[dict], dict]:
    original_hash = digest(bank)
    scoped_task = case.task.model_copy(update={"namespace": namespace})
    eligible_bank = tuple(x for x in bank if eligible(scoped_task, x))
    started = time.perf_counter()
    ids = list(
        select(Task.model_validate(scoped_task.model_dump(mode="json")), deepcopy(eligible_bank))
    )
    if digest(bank) != original_hash:
        raise RuntimeError("frozen bank mutated")
    if len(ids) != len(set(ids)) or not set(ids) <= {x["id"] for x in eligible_bank}:
        raise ValueError("selection must contain distinct IDs from frozen bank")
    selected = _bounded([x for x in eligible_bank if x["id"] in ids], budget)
    cost = {"latency_ms": (time.perf_counter() - started) * 1000} | _usage(select)
    return selected, cost


def _assert_frozen(store: Store, namespace: str, frozen: dict) -> None:
    generation, lessons = store.snapshot(namespace)
    if generation != frozen["generation"] or digest(lessons) != frozen["hash"]:
        raise RuntimeError("frozen memory changed")


def _episode(
    case: Case,
    cycle_id: str,
    result: SolveResult,
    verified: VerifyResult,
    namespace: str,
    solver_model: str,
) -> Episode:
    observations = [
        o.model_copy(update={"id": f"{case.task.id}/{o.id}"}) for o in result.observations
    ]
    observations.append(
        Observation.create(f"{case.task.id}/cycle-verifier", "verifier", verified.details)
    )
    return Episode(
        id=f"{cycle_id}/{case.task.id}",
        task_id=case.task.id,
        task_group=case.matter_id,
        namespace=namespace,
        task=case.task.text,
        host_revision=cycle_id,
        solver_model=solver_model,
        environment=case.task.environment,
        tools=case.task.tools,
        observations=observations,
        verifier_result="pass" if verified.passed else "fail",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )


def summarize(state: dict) -> dict:
    """Aggregate observed query outcomes by cohort and arm; costs stay separate."""
    result: dict = {
        cohort: {
            arm: {
                "attempts": 0,
                "graded": 0,
                "all_pass": 0,
                "criterion_pass": {},
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
                "tool_calls": 0,
                "compute": 0,
            }
            for arm in ARMS
        }
        for cohort in ("familiar", "new")
    }
    overhead = {
        "sleep": [],
        "query_gate": [],
        "support_gate": [],
        "support_solver": [],
        "support_grader": [],
        "query_grader": [],
    }
    for cycle in state["protocol"]["cycles"]:
        record = state["cycles"].get(cycle["id"], {})
        if record.get("sleep_cost"):
            overhead["sleep"].append(record["sleep_cost"])
        overhead["query_gate"].extend(record.get("gate_cost", {}).values())
        overhead["support_gate"].extend(record.get("support_gate_cost", {}).values())
        for support in record.get("support", {}).values():
            overhead["support_solver"].append(support["metrics"])
            overhead["support_grader"].extend(
                support.get("grades", {}).get("grade_cost", {}).values()
            )
        for case in cycle["query"]:
            row = record.get("queries", {}).get(case["task"]["id"], {})
            for arm in ARMS:
                if arm not in row:
                    continue
                attempt = row[arm]
                overhead["query_grader"].extend(attempt.get("grade_cost", {}).values())
                target = result[case["cohort"]][arm]
                target["attempts"] += 1
                if attempt.get("all_pass") is not None:
                    target["graded"] += 1
                    target["all_pass"] += int(attempt["all_pass"])
                for rubric, passed in (attempt.get("criterion_pass") or {}).items():
                    counts = target["criterion_pass"].setdefault(rubric, {"graded": 0, "passed": 0})
                    if passed is not None:
                        counts["graded"] += 1
                        counts["passed"] += int(passed)
                for metric in (
                    "input_tokens",
                    "output_tokens",
                    "latency_ms",
                    "tool_calls",
                    "compute",
                ):
                    target[metric] += attempt["metrics"].get(metric) or 0
    return {"cohorts": result, "overhead": overhead}


def run_cycles(
    protocol: Protocol,
    *,
    manifest_path: str | Path,
    store: Store,
    solve: Callable[[Task, str, Mapping[str, object]], SolveResult],
    verify: Callable[[Task, object], VerifyResult],
    batch_sleep: Callable[[tuple[GradedEpisode, ...], Store, str], Mapping[str, object]],
    select: Callable[[Task, tuple[dict, ...]], Sequence[str]],
    graders: Mapping[str, Callable[[Task, object, tuple[str, ...]], GradeResult]],
) -> dict:
    """Run or resume a fixed protocol; callbacks must be stable across resumption.

    Sleep is called once per cycle on support episodes. Query answers and grades
    never enter Sleep. A changed store after freezing aborts rather than silently
    evaluating different memory. An interrupted mutating Sleep requires an
    idempotent adapter or a fresh run; it is never automatically replayed.
    """
    validate_protocol(protocol)
    if set(protocol.grader_ids) != set(graders):
        raise ValueError("grader mapping must match the two declared grader IDs")
    path = Path(manifest_path)
    specification = _protocol_data(protocol)
    protocol_hash = digest(specification)
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("protocol_hash") != protocol_hash or state.get("protocol") != specification:
            raise ValueError("manifest protocol mismatch")
    else:
        state = {
            "version": 1,
            "protocol_hash": protocol_hash,
            "protocol": specification,
            "cycles": {},
            "complete": False,
        }
        _atomic_json(path, state)

    def save() -> None:
        _atomic_json(path, state)

    for cycle in protocol.cycles:
        namespace = f"{protocol.run_id}/memory"
        record = state["cycles"].setdefault(
            cycle.id,
            {
                "namespace": namespace,
                "support": {},
                "queries": {},
                "stage": "support",
                "sleep_cost": {},
                "gate_cost": {},
                "support_gate_cost": {},
            },
        )
        if record["namespace"] != namespace:
            raise ValueError("namespace mismatch")
        save()
        if record["stage"] == "complete":
            if digest(record["snapshot"]["lessons"]) != record["snapshot"]["hash"]:
                raise ValueError("frozen snapshot corrupted")
            following = protocol.cycles.index(cycle) + 1
            if (
                following == len(protocol.cycles)
                or protocol.cycles[following].id not in state["cycles"]
            ):
                _assert_frozen(store, namespace, record["snapshot"])
            continue
        if (
            not record["support"]
            and "snapshot" not in record
            and store.generation(namespace)
            != (
                0
                if cycle == protocol.cycles[0]
                else state["cycles"][protocol.cycles[protocol.cycles.index(cycle) - 1].id][
                    "snapshot"
                ]["generation"]
            )
        ):
            raise RuntimeError("memory generation differs from preceding cycle")
        for case in cycle.support:
            if case.task.id in record["support"]:
                continue
            previous = protocol.cycles.index(cycle) - 1
            prior_bank = (
                ()
                if previous < 0
                else tuple(state["cycles"][protocol.cycles[previous].id]["snapshot"]["lessons"])
            )
            if previous >= 0:
                _assert_frozen(
                    store, namespace, state["cycles"][protocol.cycles[previous].id]["snapshot"]
                )
            if prior_bank:
                selected, gate_cost = _select(
                    case, prior_bank, namespace, protocol.context_budget, select
                )
                _assert_frozen(
                    store, namespace, state["cycles"][protocol.cycles[previous].id]["snapshot"]
                )
                record["support_gate_cost"][case.task.id] = gate_cost
            else:
                selected = []
            result = solve(
                Task.model_validate(case.task.model_dump(mode="json")),
                render(selected),
                deepcopy(specification["solver_settings"]),
            )
            answer = _answer_data(deepcopy(result.answer))
            checked = verify(
                Task.model_validate(case.task.model_dump(mode="json")), deepcopy(answer)
            )
            episode = _episode(
                case,
                cycle.id,
                result,
                checked,
                namespace,
                str(protocol.solver_settings.get("model", "unspecified")),
            )
            graded = _grades(
                case, answer, graders, protocol.grader_ids, protocol.disagreement, checked.passed
            )
            if previous >= 0:
                _assert_frozen(
                    store, namespace, state["cycles"][protocol.cycles[previous].id]["snapshot"]
                )
            verdicts = [
                RubricVerdict(
                    criterion=criterion,
                    verdict=(
                        "unknown"
                        if graded["criterion_pass"][criterion] is None
                        else "pass"
                        if graded["criterion_pass"][criterion]
                        else "fail"
                    ),
                    evidence_ids=[f"{case.task.id}/cycle-verifier"],
                )
                for criterion in case.rubric_ids
            ]
            support = GradedEpisode(
                episode=episode,
                matter_id=case.matter_id,
                rubric=verdicts,
                applied_lessons={x["id"]: x["revision"] for x in selected},
                deliverables=[
                    *(result.deliverables or (canonical(answer),)),
                    *([case.reference] if case.reference else []),
                ],
            )
            record["support"][case.task.id] = {
                "graded_episode": support.model_dump(mode="json"),
                "passed": checked.passed,
                "grades": graded,
                "metrics": _metric(result),
                "selected": [{"id": x["id"], "revision": x["revision"]} for x in selected],
            }
            save()
        if "snapshot" not in record:
            previous = protocol.cycles.index(cycle) - 1
            if previous >= 0:
                _assert_frozen(
                    store, namespace, state["cycles"][protocol.cycles[previous].id]["snapshot"]
                )
            before = store.generation(namespace)
            if record["stage"] == "sleep_started" and before != record["pre_sleep_generation"]:
                raise RuntimeError("Sleep mutated memory before receipt; inspect before resuming")
            record["stage"] = "sleep_started"
            record["pre_sleep_generation"] = before
            save()
            episodes = tuple(
                GradedEpisode.model_validate(record["support"][x.task.id]["graded_episode"])
                for x in cycle.support
            )
            started = time.perf_counter()
            sleep_cost = dict(batch_sleep(episodes, store, namespace))
            sleep_cost.update(_usage(batch_sleep))
            sleep_cost.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
            generation, lessons = store.snapshot(namespace)
            record["sleep_cost"] = sleep_cost
            record["snapshot"] = {
                "generation": generation,
                "lessons": lessons,
                "hash": digest(lessons),
            }
            record["stage"] = "query"
            save()
        frozen = record["snapshot"]
        if digest(frozen["lessons"]) != frozen["hash"]:
            raise ValueError("frozen snapshot corrupted")
        _assert_frozen(store, namespace, frozen)
        bank = tuple(deepcopy(frozen["lessons"]))
        for case in cycle.query:
            row = record["queries"].setdefault(case.task.id, {})
            scoped_task = case.task.model_copy(update={"namespace": namespace})
            scoped_bank = tuple(x for x in bank if eligible(scoped_task, x))
            for arm in ARMS:
                if arm in row:
                    continue
                if arm == "no_memory":
                    chosen: list[dict] = []
                elif arm == "full_memory":
                    chosen = list(scoped_bank)
                    if token_upper_bound(render(chosen)) > protocol.context_budget:
                        raise ValueError("full_memory exceeds context budget")
                else:
                    chosen, gate_cost = _select(
                        case, bank, namespace, protocol.context_budget, select
                    )
                    record["gate_cost"][case.task.id] = gate_cost
                _assert_frozen(store, namespace, frozen)
                context = render(chosen)
                result = solve(
                    Task.model_validate(case.task.model_dump(mode="json")),
                    context,
                    deepcopy(specification["solver_settings"]),
                )
                answer = _answer_data(deepcopy(result.answer))
                checked = verify(
                    Task.model_validate(case.task.model_dump(mode="json")), deepcopy(answer)
                )
                _assert_frozen(store, namespace, frozen)
                row[arm] = {
                    "answer": answer,
                    "verifier_passed": checked.passed,
                    "selected": [{"id": x["id"], "revision": x["revision"]} for x in chosen],
                    "context_hash": digest(context),
                    "metrics": _metric(result),
                }
                save()
            # Grading is a separate phase: it never reaches solve or batch_sleep.
            for arm in ARMS:
                attempt = row[arm]
                if "grades" in attempt:
                    continue
                attempt.update(
                    _grades(
                        case,
                        attempt["answer"],
                        graders,
                        protocol.grader_ids,
                        protocol.disagreement,
                        attempt["verifier_passed"],
                    )
                )
                _assert_frozen(store, namespace, frozen)
                save()
        _assert_frozen(store, namespace, frozen)
        record["stage"] = "complete"
        save()
    state["complete"] = True
    state["summary"] = summarize(state)
    save()
    return state
