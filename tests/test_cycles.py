import json

import pytest

from jev_memory.consolidation import Consolidator, EvidenceRef, GradedEpisode, MemoryOperation
from jev_memory.cycles import (
    Case,
    Cycle,
    GradeResult,
    Protocol,
    SolveResult,
    VerifyResult,
    _grades,
    run_cycles,
    validate_protocol,
)
from jev_memory.models import Observation, Task, canonical, digest
from jev_memory.store import Store


def protocol():
    def case(name, cohort="familiar", rubric=()):
        return Case(Task(id=name, namespace="seed", text=name), name, cohort, rubric)

    return Protocol(
        cycles=(
            Cycle("c1", (case("s1", rubric=("correct",)),), (case("q1", rubric=("correct",)),)),
            Cycle("c2", (case("s2", rubric=("correct",)),), (case("q2", "new", ("correct",)),)),
        ),
        solver_settings={"model": "local", "temperature": 0, "max_tokens": 100},
        context_budget=1200,
        run_id="run1",
        grader_ids=("judge-a", "judge-b"),
    )


def test_rejects_task_and_matter_leakage():
    p = protocol()
    first = p.cycles[0]
    with pytest.raises(ValueError, match="task leakage"):
        validate_protocol(
            p.__class__(
                **{
                    **p.__dict__,
                    "cycles": (
                        Cycle(
                            first.id,
                            first.support,
                            (Case(first.support[0].task, "s1", "familiar", ("correct",)),),
                        ),
                        *p.cycles[1:],
                    ),
                }
            )
        )
    leak = Case(Task(id="other", namespace="seed", text="other"), "s1", "new", ("correct",))
    with pytest.raises(ValueError, match="matter leakage"):
        validate_protocol(
            p.__class__(
                **{**p.__dict__, "cycles": (Cycle(first.id, first.support, (leak,)), *p.cycles[1:])}
            )
        )


def test_cycle_runs_and_resumes_same_frozen_bank(tmp_path):
    calls = {"solve": 0, "sleep": 0, "grade": 0}
    contexts = []

    def solve(task, context, settings):
        calls["solve"] += 1
        contexts.append((task.id, context, dict(settings)))
        return SolveResult(
            answer={"ok": True},
            observations=(Observation.create("tool", "tool", "observed"),),
            input_tokens=3,
            output_tokens=2,
            latency_ms=4,
            tool_calls=1,
            compute=0.5,
        )

    def verify(task, answer):
        return VerifyResult(answer["ok"], "verified")

    def sleep(episodes, store, namespace):
        calls["sleep"] += 1
        assert len(episodes) == 1
        assert episodes[0].episode.verifier_result == "pass"
        assert episodes[0].rubric[0].verdict == "pass"
        assert episodes[0].deliverables
        store.episode(episodes[0].episode)
        with store.transaction():
            store.db.execute(
                "INSERT INTO revisions(namespace,id,revision,status,body,identity,validation) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    namespace,
                    f"lesson-{episodes[0].episode.task_id}",
                    1,
                    "active",
                    canonical(
                        {
                            "namespace": namespace,
                            "trigger": "x",
                            "lesson": "Remember this",
                            "preconditions": ["x"],
                            "exceptions": [],
                            "evidence": {"o": "0" * 64},
                            "required_tools": ["other"]
                            if episodes[0].episode.task_id == "s2"
                            else [],
                        }
                    ),
                    "identity",
                    "source",
                ),
            )
            store._bump(namespace)
        return {"input_tokens": 7, "output_tokens": 8, "compute": 1.2}

    def select(task, bank):
        assert task.namespace == "run1/memory"
        return [bank[0]["id"]]

    def grade(task, answer, rubrics):
        calls["grade"] += 1
        return GradeResult({"correct": answer["ok"]}, "correct")

    path = tmp_path / "manifest.json"
    with Store(tmp_path / "bank.sqlite") as store:
        kwargs = dict(
            manifest_path=path,
            store=store,
            solve=solve,
            verify=verify,
            batch_sleep=sleep,
            select=select,
            graders={"judge-a": grade, "judge-b": grade},
        )
        state = run_cycles(protocol(), **kwargs)
        assert state["complete"]
        assert calls == {"solve": 8, "sleep": 2, "grade": 16}
        for cycle in protocol().cycles:
            row = state["cycles"][cycle.id]
            assert row["snapshot"]["generation"] == (1 if cycle.id == "c1" else 2)
            assert row["sleep_cost"]["input_tokens"] == 7
            query = row["queries"][cycle.query[0].task.id]
            assert query["no_memory"]["selected"] == []
            assert query["selected_memory"]["selected"] == query["full_memory"]["selected"][:1]
            assert len(query["full_memory"]["selected"]) == 1
            assert all(query[a]["all_pass"] for a in query)
            assert query["selected_memory"]["metrics"]["tool_calls"] == 1
        assert contexts[1][1] == ""  # first query / no memory
        assert "Remember this" in contexts[4][1]  # next support uses prior memory
        assert "Remember this" in contexts[2][1]
        assert state["cycles"]["c2"]["support"]["s2"]["selected"]
        assert state["summary"]["cohorts"]["new"]["full_memory"]["all_pass"] == 1
        assert len(state["summary"]["overhead"]["sleep"]) == 2
        run_cycles(protocol(), **kwargs)
        assert calls == {"solve": 8, "sleep": 2, "grade": 16}
        with store.transaction():
            store.db.execute("UPDATE revisions SET status='archived' WHERE id=?", ("lesson-s2",))
        with pytest.raises(RuntimeError, match="frozen memory changed"):
            run_cycles(protocol(), **kwargs)
    assert json.loads(path.read_text())["complete"]


def test_grader_disagreement_is_explicit(tmp_path):
    p = protocol()
    p = Protocol(
        (p.cycles[0],), p.solver_settings, p.context_budget, p.run_id, p.grader_ids, "exclude"
    )

    def sleep(episodes, store, namespace):
        return {}

    def grade(value):
        return lambda task, answer, rubrics: GradeResult({"correct": value}, "correct")

    with Store(tmp_path / "bank.sqlite") as store:
        state = run_cycles(
            p,
            manifest_path=tmp_path / "manifest.json",
            store=store,
            solve=lambda task, context, settings: SolveResult(answer="answer"),
            verify=lambda task, answer: VerifyResult(True, "pass"),
            batch_sleep=sleep,
            select=lambda task, bank: [],
            graders={"judge-a": grade(True), "judge-b": grade(False)},
        )
    query = state["cycles"]["c1"]["queries"]["q1"]
    assert all(query[arm]["all_pass"] is None for arm in query)
    assert all(query[arm]["grader_agreement"] is False for arm in query)


def test_failed_verifier_cannot_be_excluded_by_grader_disagreement():
    case = protocol().cycles[0].query[0]
    graders = {
        "judge-a": lambda task, answer, rubrics: GradeResult({"correct": True}, "correct"),
        "judge-b": lambda task, answer, rubrics: GradeResult({"correct": False}, "correct"),
    }
    result = _grades(case, "answer", graders, ("judge-a", "judge-b"), "exclude", False)
    assert result["criterion_pass"] == {"correct": None}
    assert result["all_pass"] is False


def test_legacy_archived_default_identity_cannot_be_readded(store, gate, episode, candidate):
    legacy_identity = digest(candidate.model_dump(exclude={"evidence", "kind", "work_kind"}))
    assert candidate.identity == legacy_identity
    assert candidate.model_copy(update={"kind": "checklist"}).identity != legacy_identity
    with store.transaction():
        store.db.execute(
            "INSERT INTO revisions(namespace,id,revision,status,body,identity,validation) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                "demo",
                legacy_identity,
                1,
                "archived",
                canonical(candidate),
                legacy_identity,
                "source",
            ),
        )
    graded = [GradedEpisode(episode=episode, matter_id="matter-1")]
    refs = [
        EvidenceRef(
            episode_id=episode.id,
            observation_id=episode.observations[0].id,
            sha256=episode.observations[0].sha256,
        )
    ]
    operation = MemoryOperation(action="add", candidate=candidate, evidence=refs)
    with pytest.raises(ValueError, match="archived_identity"):
        Consolidator(store, gate, shadow=False)._validate([operation], graded, [])


def test_boolean_grades_and_full_bank_budget_are_enforced(tmp_path):
    p = protocol()
    p = Protocol((p.cycles[0],), p.solver_settings, 1, p.run_id, p.grader_ids)
    with Store(tmp_path / "bank.sqlite") as store:

        def grade(value):
            return lambda task, answer, rubrics: GradeResult({"correct": value}, "correct")

        def sleep(episodes, store, namespace):
            store.episode(episodes[0].episode)
            with store.transaction():
                store.db.execute(
                    "INSERT INTO revisions(namespace,id,revision,status,body,identity,validation) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        namespace,
                        "one",
                        1,
                        "active",
                        canonical(
                            {
                                "namespace": namespace,
                                "trigger": "x",
                                "lesson": "A lesson",
                                "preconditions": ["x"],
                                "exceptions": [],
                                "evidence": {"o": "0" * 64},
                            }
                        ),
                        "identity",
                        "source",
                    ),
                )
                store._bump(namespace)
            return {}

        common = dict(
            manifest_path=tmp_path / "manifest.json",
            store=store,
            solve=lambda task, context, settings: SolveResult("answer"),
            verify=lambda task, answer: VerifyResult(True, "pass"),
            batch_sleep=sleep,
            select=lambda task, bank: [],
            graders={"judge-a": grade(True), "judge-b": grade(True)},
        )
        with pytest.raises(ValueError, match="full_memory exceeds"):
            run_cycles(p, **common)
        assert (
            "no_memory"
            in json.loads((tmp_path / "manifest.json").read_text())["cycles"]["c1"]["queries"]["q1"]
        )

    with Store(tmp_path / "other.sqlite") as store:
        bad = dict(
            common,
            manifest_path=tmp_path / "other.json",
            store=store,
            batch_sleep=lambda episodes, store, namespace: {},
            graders={"judge-a": grade(1), "judge-b": grade(True)},
        )
        with pytest.raises(ValueError, match="booleans"):
            run_cycles(p, **bad)
