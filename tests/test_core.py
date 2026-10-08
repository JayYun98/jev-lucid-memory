import json

import pytest
from pydantic import ValidationError

from jev_memory.core import Memory, token_upper_bound
from jev_memory.models import Decision, Observation, Policy
from jev_memory.store import Conflict, Store


def test_sleep_admit_wake_persist(store, gate, episode, candidate, task):
    m = Memory(store, gate, shadow=False)
    result = m.admit_lesson(episode, candidate)
    assert result["status"] == "active"
    wake = m.wake(task)
    assert wake["selected"] == [{"id": result["id"], "revision": 1}]
    assert candidate.lesson in wake["context"]
    assert wake["token_upper_bound"] <= 1200
    with Store(store.db.execute("PRAGMA database_list").fetchone()[2]) as reopened:
        assert reopened.lessons("demo")[0]["id"] == result["id"]
    audit = store.events("demo")
    assert any(e["body"].get("instruction_hash") for e in audit)


def test_shadow_does_not_activate_or_inject(store, gate, episode, candidate, task):
    m = Memory(store, gate)
    assert m.admit_lesson(episode, candidate)["status"] == "active"
    assert store.lessons("demo") == []
    Memory(store, gate, shadow=False).admit_lesson(episode, candidate)
    assert m.wake(task)["context"] == ""
    assert m.wake(task)["selected"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("namespace", "other", "namespace_mismatch"),
        ("evidence", {"missing": "0" * 64}, "invalid_evidence"),
        ("required_tools", ["shell"], "unobserved_tool_requirement"),
        ("environment", {"api": "v2"}, "unobserved_environment"),
    ],
)
def test_deterministic_rejection(store, gate, episode, candidate, field, value, reason):
    result = Memory(store, gate, shadow=False).admit_lesson(
        episode, candidate.model_copy(update={field: value})
    )
    assert result["reason"] == reason
    assert not gate.calls


def test_assistant_is_not_evidence(store, gate, episode, candidate):
    o = Observation.create("claim", "assistant", "I fixed everything")
    ep = episode.model_copy(update={"observations": [o], "verifier_result": "unknown"})
    c = candidate.model_copy(update={"evidence": {o.id: o.sha256}})
    assert Memory(store, gate).admit_lesson(ep, c)["reason"] == "assistant_only_evidence"


def test_evidence_hash_tamper():
    with pytest.raises(ValidationError):
        Observation(id="x", kind="tool", content="abc", sha256="0" * 64)


def test_episode_identity_cannot_change(store, episode):
    store.episode(episode)
    with pytest.raises(Conflict):
        store.episode(episode.model_copy(update={"task": "different"}))


def test_idempotent_delivery(store, gate, episode, candidate):
    m = Memory(store, gate, shadow=False)
    first = m.admit_lesson(episode, candidate)
    calls = len(gate.calls)
    assert m.admit_lesson(episode, candidate)["id"] == first["id"]
    assert len(gate.calls) == calls
    assert store.db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 2


def test_duplicate_merges_evidence(store, gate, episode, candidate):
    m = Memory(store, gate, shadow=False)
    first = m.admit_lesson(episode, candidate)
    gate.scores["duplicate_0"] = 0.99
    ep = episode.model_copy(update={"id": "episode-2", "task_id": "task-2"})
    second = m.admit_lesson(ep, candidate)
    assert second["id"] == first["id"] and second["revision"] == 1 and second["duplicate"]
    assert len(store.lessons("demo")) == 1
    assert store.db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 4


@pytest.mark.parametrize("score,status", [(0.1, "rejected"), (0.5, "pending"), (0.9, "active")])
def test_admission_thresholds(store, gate, episode, candidate, score, status):
    gate.scores["supported"] = score
    result = Memory(store, gate, shadow=False).admit_lesson(episode, candidate)
    assert result["status"] == status
    assert bool(store.lessons("demo")) == (status == "active")


def test_contradiction_pending(store, gate, episode, candidate):
    m = Memory(store, gate, shadow=False)
    m.admit_lesson(episode, candidate)
    gate.scores["contradiction_0"] = 0.95
    c = candidate.model_copy(update={"lesson": "Always stop at page one."})
    assert m.admit_lesson(episode, c)["status"] == "pending"
    assert len(store.lessons("demo")) == 1


def test_failure_pending_and_retryable(store, gate, episode, candidate):
    class Broken:
        model = "broken"

        def decide(self, *args):
            raise TimeoutError("SECRET")

    assert (
        Memory(store, Broken(), shadow=False).admit_lesson(episode, candidate)["status"]
        == "pending"
    )
    assert "SECRET" not in json.dumps(store.events("demo"))
    assert Memory(store, gate, shadow=False).admit_lesson(episode, candidate)["status"] == "active"


def test_revision_and_archive(store, gate, episode, candidate, task):
    m = Memory(store, gate, shadow=False)
    first = m.admit_lesson(episode, candidate)
    new = candidate.model_copy(
        update={"lesson": "Follow next_cursor to null; accumulate each returned record."}
    )
    result = m.admit_lesson(episode, new, replace_id=first["id"], expected_revision=1)
    assert result["revision"] == 2
    with pytest.raises(Conflict):
        store.transition("demo", first["id"], 1, "archived")
    store.transition("demo", first["id"], 2, "archived")
    assert m.wake(task)["context"] == ""
    assert store.db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 3


def test_concurrent_writer_rejects_stale_assessment(store, gate, episode, candidate):
    m = Memory(store, gate, shadow=False)
    first = m.admit_lesson(episode, candidate)

    class Racing:
        model = "racing"

        def decide(self, state, questions):
            store.transition("demo", first["id"], 1, "archived")
            return gate.decide(state, questions)

    new = candidate.model_copy(update={"lesson": "Another lesson"})
    with pytest.raises(Conflict):
        Memory(store, Racing(), shadow=False).admit_lesson(episode, new)


@pytest.mark.parametrize(
    "update", [{"namespace": "other"}, {"tools": []}, {"environment": {"api": "v2"}}]
)
def test_scope_filters_before_model_call(store, gate, episode, candidate, task, update):
    m = Memory(store, gate, shadow=False)
    m.admit_lesson(episode, candidate)
    count = len(gate.calls)
    assert not m.wake(task.model_copy(update=update))["selected"]
    assert len(gate.calls) == count


def test_retrieval_abstains(store, gate, episode, candidate, task):
    m = Memory(store, gate, shadow=False)
    m.admit_lesson(episode, candidate)
    gate.scores["applicable"] = 0.05
    assert (
        m.wake(task.model_copy(update={"text": "Fetch only first page of cursor pagination"}))[
            "selected"
        ]
        == []
    )


def test_budget_includes_wrapper_and_non_ascii(store, gate, episode, candidate, task):
    m = Memory(store, gate, Policy(token_budget=10), shadow=False)
    m.admit_lesson(episode, candidate)
    assert not m.wake(task)["context"]
    assert token_upper_bound("한글") == 6


def test_invalid_noul():
    for score in (float("nan"), float("inf"), -1, 2):
        with pytest.raises(ValidationError):
            Decision(model="x", scores={"a": score})


def test_reflection_lifecycle(store, gate, episode, candidate):
    from jev_memory.models import Usage

    class Writer:
        model = "test-writer"

        def propose(self, ep):
            return [candidate], Usage(input_tokens=20)

    m = Memory(store, gate, shadow=False)
    assert m.sleep(episode, Writer())["results"][0]["status"] == "active"
    gate.scores["reflect"] = 0.01
    assert not m.sleep(episode, Writer())["reflected"]

    class BrokenWriter:
        model = "broken"

        def propose(self, ep):
            raise ValueError("secret")

    assert m.sleep(episode, BrokenWriter(), trigger=False)["error"] == "reflection_failed"


def test_draft_cannot_choose_namespace(episode, candidate):
    from jev_memory.models import Draft

    body = candidate.model_dump(exclude={"namespace", "evidence"}) | {
        "evidence_ids": list(candidate.evidence)
    }
    draft = Draft.model_validate(body)
    assert draft.resolve(episode).namespace == episode.namespace
    assert draft.resolve(episode).evidence == candidate.evidence
    with pytest.raises(ValidationError):
        Draft.model_validate(body | {"namespace": "attacker"})
    with pytest.raises(ValueError):
        draft.model_copy(update={"evidence_ids": ["unknown"]}).resolve(episode)


def test_exact_content_duplicate_keeps_revision(store, gate, episode, candidate):
    from jev_memory.core import Memory

    memory = Memory(store, gate, shadow=False)
    first = memory.admit_lesson(episode, candidate)
    second_episode = episode.model_copy(update={"id": "episode-2", "task_id": "task-2"})
    second = memory.admit_lesson(second_episode, candidate)
    assert second["id"] == first["id"]
    assert second["revision"] == 1
    assert second["duplicate"] is True


def test_gate_thresholds_can_be_calibrated_separately(store, episode, candidate, task):
    from conftest import ScriptedGate

    from jev_memory.core import Memory
    from jev_memory.models import Policy

    gate = ScriptedGate(supported=0.7, transferable=0.7, actionable=0.7, applicable=0.7)
    memory = Memory(
        store, gate, Policy(admission_threshold=0.65, retrieval_threshold=0.9), shadow=False
    )
    assert memory.admit_lesson(episode, candidate)["status"] == "active"
    assert memory.wake(task)["selected"] == []
