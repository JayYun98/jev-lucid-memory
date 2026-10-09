import pytest

from jev_memory.consolidation import (
    ConsolidationPolicy,
    Consolidator,
    EvidenceRef,
    GradedEpisode,
    MemoryOperation,
    RubricVerdict,
)
from jev_memory.models import Observation, canonical
from jev_memory.providers import ProviderError
from jev_memory.store import Conflict


class Reviewer:
    model = "test-reviewer"

    def __init__(self, operations):
        self.operations = operations
        self.calls = 0

    def propose(self, episodes, snapshot):
        self.calls += 1
        return self.operations


def samples(episode, candidate):
    second_obs = Observation.create(
        "tool-2", "tool", "Second matter independently confirmed cursor traversal."
    )
    ep2 = episode.model_copy(
        update={
            "id": "episode-2",
            "task_id": "task-2",
            "observations": [second_obs],
            "verifier_result": "unknown",
        }
    )
    graded = [
        GradedEpisode(episode=episode, matter_id="matter-1"),
        GradedEpisode(episode=ep2, matter_id="matter-2"),
    ]
    refs = [
        EvidenceRef(
            episode_id=episode.id,
            observation_id=episode.observations[0].id,
            sha256=episode.observations[0].sha256,
        ),
        EvidenceRef(episode_id=ep2.id, observation_id=second_obs.id, sha256=second_obs.sha256),
    ]
    c = candidate.model_copy(update={"evidence": {r.observation_id: r.sha256 for r in refs}})
    return graded, refs, c


def test_batch_add_and_shadow_and_idempotency(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    reviewer = Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)])
    assert Consolidator(store, gate).run(graded, reviewer, "cycle-1")["status"] == "shadow"
    assert store.lessons("demo") == []
    live = Consolidator(store, gate, shadow=False)
    result = live.run(graded, reviewer, "cycle-1")
    assert result["status"] == "active"
    assert len(store.lessons("demo")) == 1
    assert store.db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 2
    calls = reviewer.calls
    assert live.run(graded, reviewer, "cycle-1") == result
    assert reviewer.calls == calls
    assert live.queue("demo") == []


def test_singleton_and_repeated_observation_queue(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    reviewer = Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs[:1])])
    result = live.run(graded, reviewer, "single")
    assert result["status"] == "pending"
    queued = live.queue("demo")[0]
    assert queued["reason"] == "insufficient_independent_support"
    assert queued["payload"]["operations"][0]["candidate"]["lesson"] == c.lesson
    assert queued["payload"]["review"][0]["reason"] == "insufficient_independent_support"
    assert not store.lessons("demo")
    reviewer.operations = [MemoryOperation(action="add", candidate=c, evidence=refs)]
    assert live.retry(result["queue_id"], reviewer)["status"] == "active"
    assert live.queue("demo", status="resolved")


def test_invalid_batch_does_not_partially_commit(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    invalid = c.model_copy(update={"evidence": {"missing": "0" * 64}})
    reviewer = Reviewer(
        [
            MemoryOperation(action="add", candidate=c, evidence=refs),
            MemoryOperation(action="add", candidate=invalid, evidence=refs),
        ]
    )
    result = Consolidator(store, gate, shadow=False).run(graded, reviewer, "invalid")
    assert result["status"] == "rejected"
    assert result["reason"] == "unbound_candidate_evidence"
    assert not store.lessons("demo")


def test_stale_revision_cannot_mutate(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    first = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "add"
    )
    memory_id = first["operations"][0]["id"]
    store.transition("demo", memory_id, 1, "archived")
    changed = c.model_copy(update={"lesson": "Revised advice"})
    op = MemoryOperation(
        action="revise",
        candidate=changed,
        evidence=refs,
        target_ids=[memory_id],
        expected_revisions={memory_id: 1},
    )
    with pytest.raises(Conflict):
        live.run(graded, Reviewer([op]), "stale")
    assert store.lessons("demo") == []


def test_full_bank_compared_in_bounded_batches(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    with store.transaction():
        for index in range(123):
            old = c.model_copy(update={"lesson": f"Existing advice {index}"})
            store.db.execute(
                "INSERT INTO revisions(namespace,id,revision,status,body,identity) VALUES (?,?,?,?,?,?)",
                ("demo", f"old-{index}", 1, "active", canonical(old), old.identity),
            )
        store._bump("demo")
    live = Consolidator(store, gate, ConsolidationPolicy(comparison_batch=32), shadow=False)
    result = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "big"
    )
    assert result["status"] == "active"
    comparison_calls = [state for state, questions in gate.calls if "related" in state]
    assert len(comparison_calls) == 4
    assert sum(len(state["related"]) for state in comparison_calls) == 123


def test_archive_needs_cited_failure(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    first = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "add"
    )
    memory_id = first["operations"][0]["id"]
    graded[0] = graded[0].model_copy(update={"applied_lessons": {memory_id: 1}})
    op = MemoryOperation(
        action="archive", evidence=refs, target_ids=[memory_id], expected_revisions={memory_id: 1}
    )
    assert live.run(graded, Reviewer([op]), "archive-no-failure")["status"] == "rejected"
    graded[0] = graded[0].model_copy(
        update={
            "rubric": [
                RubricVerdict(
                    criterion="Completeness", verdict="fail", evidence_ids=[refs[0].observation_id]
                )
            ]
        }
    )
    assert live.run(graded, Reviewer([op]), "archive-failed")["status"] == "active"
    assert store.lessons("demo") == []


def test_revision_replaces_search_and_preserves_history(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    first = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "first"
    )
    memory_id = first["operations"][0]["id"]
    revised = c.model_copy(update={"lesson": "Revised verified cursor guidance."})
    op = MemoryOperation(
        action="revise",
        candidate=revised,
        evidence=refs,
        target_ids=[memory_id],
        expected_revisions={memory_id: 1},
    )
    result = live.run(graded, Reviewer([op]), "second")
    assert result["operations"][0]["revision"] == 2
    assert len(store.lessons("demo", active=False)) == 1
    assert (
        store.db.execute("SELECT count(*) FROM revisions WHERE id=?", (memory_id,)).fetchone()[0]
        == 2
    )
    assert (
        store.db.execute("SELECT count(*) FROM search WHERE id=?", (memory_id,)).fetchone()[0] == 1
    )
    assert (
        store.db.execute("SELECT text FROM search WHERE id=?", (memory_id,))
        .fetchone()[0]
        .find("Revised")
        >= 0
    )


def test_provider_error_is_queued_and_retry_rechecks(store, gate, episode, candidate):
    graded, refs, c = samples(episode, candidate)
    reviewer = Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)])

    class Broken:
        model = "broken"

        def decide(self, state, questions):
            raise TimeoutError("private")

    live = Consolidator(store, Broken(), shadow=False)
    result = live.run(graded, reviewer, "retry")
    assert result["status"] == "pending"
    assert live.queue("demo", status="pending")[0]["reason"] == "provider_error"
    queued = live.queue("demo", status="pending")[0]
    assert queued["payload"]["operations"][0]["candidate"]["lesson"] == c.lesson
    assert queued["payload"]["review"][0]["decisions"][0]["scores"] is None
    assert not store.lessons("demo")
    gate.scores["client_specific"] = 0.01
    assert (
        Consolidator(store, gate, shadow=False).retry(result["queue_id"], reviewer)["status"]
        == "active"
    )
    assert reviewer.calls == 2


def test_assistant_evidence_cannot_supply_second_task(store, gate, episode, candidate):
    graded, refs, c = samples(episode, candidate)
    claim = Observation.create("claim", "assistant", "I verified the solution")
    ep = graded[1].episode.model_copy(update={"observations": [claim]})
    graded[1] = graded[1].model_copy(update={"episode": ep})
    refs[1] = EvidenceRef(episode_id=ep.id, observation_id=claim.id, sha256=claim.sha256)
    c = c.model_copy(update={"evidence": {r.observation_id: r.sha256 for r in refs}})
    result = Consolidator(store, gate, shadow=False).run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "claim"
    )
    assert result["status"] == "pending"
    assert result["reason"] == "insufficient_independent_support"
    assert not store.lessons("demo")


def test_merge_archives_source_and_carries_provenance(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    first = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "one"
    )
    first_id = first["operations"][0]["id"]
    other = c.model_copy(update={"lesson": "Follow each next cursor."})
    second = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=other, evidence=refs)]), "two"
    )
    second_id = second["operations"][0]["id"]
    merged = c.model_copy(update={"lesson": "Follow each next cursor until it is null."})
    op = MemoryOperation(
        action="merge",
        candidate=merged,
        evidence=refs,
        target_ids=[first_id, second_id],
        expected_revisions={first_id: 1, second_id: 1},
    )
    result = live.run(graded, Reviewer([op]), "merge")
    assert result["status"] == "active"
    assert [r["id"] for r in store.lessons("demo")] == [first_id]
    assert (
        store.db.execute(
            "SELECT status FROM revisions WHERE id=? AND revision=2", (second_id,)
        ).fetchone()[0]
        == "archived"
    )
    assert (
        store.db.execute("SELECT count(*) FROM search WHERE id=?", (second_id,)).fetchone()[0] == 0
    )
    assert (
        store.db.execute("SELECT count(*) FROM evidence WHERE lesson_id=?", (first_id,)).fetchone()[
            0
        ]
        == 2
    )


def test_qualified_evidence_handles_reused_observation_ids(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    second = Observation.create("tool-1", "tool", "Independent second observation")
    graded[1] = graded[1].model_copy(
        update={"episode": graded[1].episode.model_copy(update={"observations": [second]})}
    )
    refs[1] = EvidenceRef(episode_id="episode-2", observation_id="tool-1", sha256=second.sha256)
    qualified = {f"{r.episode_id}/{r.observation_id}": r.sha256 for r in refs}
    c = c.model_copy(update={"evidence": qualified})
    result = Consolidator(store, gate, shadow=False).run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "qualified"
    )
    assert result["status"] == "active"


def test_reviewer_provider_failure_is_pending_and_retryable(store, gate, episode, candidate):
    graded, refs, c = samples(episode, candidate)

    class Unavailable:
        model = "unavailable"

        def propose(self, episodes, snapshot):
            raise ProviderError("timeout")

    live = Consolidator(store, gate, shadow=False)
    result = live.run(graded, Unavailable(), "writer-down")
    assert result["status"] == "pending"
    assert live.queue("demo", status="pending")[0]["reason"] == "provider_error"
    gate.scores["client_specific"] = 0.01
    reviewer = Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)])
    assert live.retry(result["queue_id"], reviewer)["status"] == "active"


def test_archived_identity_cannot_be_readded(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    add = MemoryOperation(action="add", candidate=c, evidence=refs)
    first = live.run(graded, Reviewer([add]), "initial")
    memory_id = first["operations"][0]["id"]
    graded[0] = graded[0].model_copy(update={"applied_lessons": {memory_id: 1}})
    failed = RubricVerdict(
        criterion="Completeness", verdict="fail", evidence_ids=[refs[0].observation_id]
    )
    graded[0] = graded[0].model_copy(update={"rubric": [failed]})
    archive = MemoryOperation(
        action="archive", evidence=refs, target_ids=[memory_id], expected_revisions={memory_id: 1}
    )
    assert live.run(graded, Reviewer([archive]), "archive")["status"] == "active"
    result = live.run(graded, Reviewer([add]), "try-readd")
    assert result["status"] == "rejected"
    assert result["reason"] == "archived_identity"
    assert (
        live.queue("demo", status="rejected")[0]["payload"]["operations"][0]["candidate"]["lesson"]
        == c.lesson
    )
    assert store.lessons("demo") == []
    assert (
        store.db.execute("SELECT count(*) FROM revisions WHERE id=?", (memory_id,)).fetchone()[0]
        == 2
    )


def test_pending_cycle_is_single_queue_item_and_input_is_immutable(store, gate, episode, candidate):
    graded, refs, c = samples(episode, candidate)

    class Broken:
        model = "broken"

        def decide(self, state, questions):
            raise TimeoutError("transient")

    reviewer = Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)])
    pending = Consolidator(store, Broken(), shadow=False)
    first = pending.run(graded, reviewer, "same-cycle")
    again = pending.run(graded, reviewer, "same-cycle")
    assert first["queue_id"] == again["queue_id"]
    assert len(pending.queue("demo")) == 1
    existing_payload = pending.queue("demo")[0]["payload"]
    with store.transaction():
        store.db.execute(
            "INSERT INTO consolidation_queue(namespace,cycle_id,status,reason,payload) "
            "VALUES (?,?,?,?,?)",
            ("demo", "same-cycle", "pending", "legacy_duplicate", canonical(existing_payload)),
        )
    changed = graded.copy()
    changed[0] = changed[0].model_copy(update={"matter_id": "changed"})
    with pytest.raises(Conflict):
        pending.run(changed, reviewer, "same-cycle")
    gate.scores["client_specific"] = 0.01
    live = Consolidator(store, gate, shadow=False)
    assert live.retry(first["queue_id"], reviewer)["status"] == "active"
    assert len(live.queue("demo", status="pending")) == 0
    assert len(live.queue("demo", status="resolved")) == 2


def test_revise_and_archive_reject_multiple_targets(store, gate, episode, candidate):
    graded, refs, c = samples(episode, candidate)
    for action in ("revise", "archive"):
        with pytest.raises(ValueError, match="invalid targets"):
            MemoryOperation(
                action=action,
                candidate=c if action == "revise" else None,
                evidence=refs,
                target_ids=["a", "b"],
                expected_revisions={"a": 1, "b": 1},
            )


def test_duplicate_adds_in_one_cycle_are_queued_without_partial_commit(
    store, gate, episode, candidate
):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    op = MemoryOperation(action="add", candidate=c, evidence=refs)
    live = Consolidator(store, gate, shadow=False)
    result = live.run(graded, Reviewer([op, op]), "duplicate-ops")
    assert result["status"] == "pending"
    assert result["reason"] == "exact_duplicate"
    assert store.lessons("demo") == []
    review = live.queue("demo")[0]["payload"]["review"]
    assert [r["status"] for r in review] == ["active", "pending"]
    assert review[0]["decisions"][0]["scores"]["supported"] > 0.8


def test_archive_gate_sees_target_and_requires_causal_retirement(store, gate, episode, candidate):
    gate.scores["client_specific"] = 0.01
    graded, refs, c = samples(episode, candidate)
    live = Consolidator(store, gate, shadow=False)
    first = live.run(
        graded, Reviewer([MemoryOperation(action="add", candidate=c, evidence=refs)]), "seed"
    )
    memory_id = first["operations"][0]["id"]
    graded[0] = graded[0].model_copy(
        update={
            "rubric": [
                RubricVerdict(
                    criterion="Unrelated formatting",
                    verdict="fail",
                    evidence_ids=[refs[0].observation_id],
                )
            ],
            "applied_lessons": {memory_id: 1},
        }
    )
    op = MemoryOperation(
        action="archive", evidence=refs, target_ids=[memory_id], expected_revisions={memory_id: 1}
    )
    gate.calls.clear()
    gate.scores["retirement_justified"] = 0.01
    result = live.run(graded, Reviewer([op]), "unrelated-failure")
    assert result["status"] == "rejected"
    assert result["reason"] == "retirement_not_justified"
    assert len(gate.calls) == 1
    state, questions = gate.calls[0]
    assert set(questions) == {"retirement_justified"}
    assert state["prior_versions"][0]["id"] == memory_id
    assert state["prior_versions"][0]["body"]["lesson"] == c.lesson
    assert state["support"][0]["applied_lessons"][memory_id] == 1
    assert store.lessons("demo")[0]["id"] == memory_id
