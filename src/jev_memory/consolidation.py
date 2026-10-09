"""Batch WakeSleep consolidation with evidence-bound, atomic memory operations."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from .models import Candidate, Episode, Identifier, Record, Text, canonical, digest
from .providers import Gate, ProviderError
from .store import Conflict, Store

VALIDATION_REASONS = {
    "archived_identity": "archived_identity",
    "insufficient independent support": "insufficient_independent_support",
    "repeated observation is not independent support": "repeated_observation",
    "assistant-only evidence": "assistant_only_evidence",
    "unknown evidence episode": "unknown_evidence_episode",
    "invalid evidence hash": "invalid_evidence_hash",
    "duplicate evidence reference": "duplicate_evidence_reference",
    "candidate evidence must bind every reference": "unbound_candidate_evidence",
    "archive requires cited failed rubric evidence": "missing_failed_rubric_evidence",
    "unobserved tool requirement": "unobserved_tool_requirement",
    "unobserved environment": "unobserved_environment",
    "namespace mismatch": "namespace_mismatch",
    "candidate kind and work kind must match operation": "kind_mismatch",
    "target modified twice in one batch": "duplicate_target",
}


class RubricVerdict(Record):
    criterion: Text
    verdict: Literal["pass", "fail", "unknown"]
    evidence_ids: list[Identifier] = Field(default_factory=list)


class GradedEpisode(Record):
    episode: Episode
    matter_id: Identifier
    rubric: list[RubricVerdict] = Field(default_factory=list)
    deliverables: list[Text] = Field(default_factory=list)
    applied_lessons: dict[Identifier, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_rubric(self):
        ids = {o.id for o in self.episode.observations}
        if any(not set(r.evidence_ids) <= ids for r in self.rubric):
            raise ValueError("rubric cites unknown observation")
        if any(revision < 1 for revision in self.applied_lessons.values()):
            raise ValueError("applied lesson revisions must be positive")
        return self


class EvidenceRef(Record):
    episode_id: Identifier
    observation_id: Identifier
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MemoryOperation(Record):
    action: Literal["add", "revise", "merge", "archive"]
    kind: Literal["checklist", "practice"] = "practice"
    work_kind: Literal["analysis", "drafting", "review", "general"] = "general"
    candidate: Candidate | None = None
    evidence: list[EvidenceRef] = Field(min_length=1)
    target_ids: list[Identifier] = Field(default_factory=list)
    expected_revisions: dict[Identifier, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def shape(self):
        if self.action == "add":
            if self.candidate is None or self.target_ids or self.expected_revisions:
                raise ValueError("add requires candidate without targets")
        else:
            required = 2 if self.action == "merge" else 1
            invalid_count = (
                len(self.target_ids) < required
                if self.action == "merge"
                else len(self.target_ids) != required
            )
            if invalid_count or len(set(self.target_ids)) != len(self.target_ids):
                raise ValueError("invalid targets")
            if set(self.expected_revisions) != set(self.target_ids):
                raise ValueError("expected revisions required for every target")
            if any(v < 1 for v in self.expected_revisions.values()):
                raise ValueError("invalid expected revision")
            if (self.action == "archive") != (self.candidate is None):
                raise ValueError("archive has no candidate; revise and merge require one")
        return self


class ConsolidationPolicy(Record):
    threshold: float = Field(default=0.8, gt=0.5, le=1)
    negative_threshold: float = Field(default=0.2, ge=0, lt=0.5)
    min_tasks: int = Field(default=2, ge=1)
    min_matters: int = Field(default=2, ge=1)
    comparison_batch: int = Field(default=32, ge=1, le=128)


class Consolidator:
    """Reviewer proposes operations; the host validates, gates and commits a whole cycle."""

    def __init__(
        self, store: Store, gate: Gate, policy: ConsolidationPolicy | None = None, *, shadow=True
    ):
        self.store, self.gate, self.policy, self.shadow = (
            store,
            gate,
            policy or ConsolidationPolicy(),
            shadow,
        )
        self.store.db.execute("""CREATE TABLE IF NOT EXISTS consolidation_queue (
            id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, cycle_id TEXT NOT NULL,
            status TEXT NOT NULL, reason TEXT NOT NULL, payload TEXT NOT NULL,
            created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))
        """)
        self.store.db.execute("""CREATE TABLE IF NOT EXISTS consolidation_cycles (
            namespace TEXT NOT NULL, cycle_id TEXT NOT NULL, input_hash TEXT NOT NULL,
            result TEXT NOT NULL, PRIMARY KEY(namespace,cycle_id))
        """)
        self.store.db.commit()

    def queue(self, namespace: str, *, status: str | None = None):
        sql = "SELECT * FROM consolidation_queue WHERE namespace=?"
        args: list = [namespace]
        if status:
            sql += " AND status=?"
            args.append(status)
        return [
            dict(r) | {"payload": json.loads(r["payload"])}
            for r in self.store.db.execute(sql + " ORDER BY id", args)
        ]

    def retry(self, queue_id: int, reviewer):
        if self.shadow:
            raise ValueError("queue retry requires shadow=False")
        row = self.store.db.execute(
            "SELECT * FROM consolidation_queue WHERE id=?", (queue_id,)
        ).fetchone()
        if row is None:
            raise KeyError(queue_id)
        payload = json.loads(row["payload"])
        episodes = [GradedEpisode.model_validate(e) for e in payload["episodes"]]
        result = self.run(episodes, reviewer, row["cycle_id"], _retry_id=queue_id)
        return result

    def _enqueue(
        self,
        namespace,
        cycle_id,
        status,
        reason,
        episodes,
        *,
        operations=None,
        review=None,
        retry_id=None,
    ):
        if self.shadow:
            return None
        episode_rows = [e.model_dump(mode="json") for e in episodes]
        payload = canonical(
            {
                "input_hash": digest(episode_rows),
                "episodes": episode_rows,
                "operations": [o.model_dump(mode="json") for o in (operations or [])],
                "review": review or [],
            }
        )
        with self.store.transaction():
            rows = self.store.db.execute(
                "SELECT id,payload FROM consolidation_queue WHERE namespace=? AND cycle_id=? ORDER BY id",
                (namespace, cycle_id),
            ).fetchall()
            if any(
                digest(json.loads(row["payload"])["episodes"]) != digest(episode_rows)
                for row in rows
            ):
                raise Conflict("cycle ID reused with different episodes")
            if rows:
                self.store.db.execute(
                    "UPDATE consolidation_queue SET status=?,reason=?,payload=? "
                    "WHERE namespace=? AND cycle_id=?",
                    (status, reason, payload, namespace, cycle_id),
                )
                return rows[0]["id"]
            cur = self.store.db.execute(
                "INSERT INTO consolidation_queue(namespace,cycle_id,status,reason,payload) VALUES (?,?,?,?,?)",
                (namespace, cycle_id, status, reason, payload),
            )
            return cur.lastrowid

    def _decision(self, namespace, state, questions):
        started = time.perf_counter()
        try:
            decision = self.gate.decide(state, questions)
            if set(decision.scores) != set(questions):
                raise ValueError("incomplete gate decision")
            self.store.event(
                namespace,
                "consolidation_gate",
                {
                    "status": "ok",
                    "model": decision.model,
                    "state_hash": digest(state),
                    "questions_hash": digest(questions),
                    "scores": decision.scores,
                    "usage": decision.usage.model_dump(mode="json"),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )
            return decision.scores
        except Exception as exc:
            self.store.event(
                namespace,
                "consolidation_gate",
                {
                    "status": "error",
                    "error": type(exc).__name__,
                    "state_hash": digest(state),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )
            return None

    def _validate(self, operations, episodes, snapshot):
        by_episode = {}
        for graded in episodes:
            ep = graded.episode
            if (
                ep.id in by_episode
                or not Episode.model_validate(ep.model_dump()).trace_hash == ep.trace_hash
            ):
                raise ValueError("invalid or duplicate episode")
            by_episode[ep.id] = graded
        by_memory = {r["id"]: r for r in snapshot}
        archived = {
            row[0]
            for row in self.store.db.execute(
                "SELECT identity FROM revisions WHERE namespace=? AND status='archived'",
                (episodes[0].episode.namespace,),
            )
        }
        touched = set()
        for op in operations:
            if op.action == "add" and op.candidate.identity in archived:
                raise ValueError("archived_identity")
            if op.candidate and op.candidate.namespace != episodes[0].episode.namespace:
                raise ValueError("namespace mismatch")
            if op.candidate and (
                op.candidate.kind != op.kind or op.candidate.work_kind != op.work_kind
            ):
                raise ValueError("candidate kind and work kind must match operation")
            for target in op.target_ids:
                row = by_memory.get(target)
                if (
                    row is None
                    or row["revision"] != op.expected_revisions[target]
                    or row["status"] != "active"
                ):
                    raise Conflict("revision mismatch")
                if target in touched:
                    raise ValueError("target modified twice in one batch")
                touched.add(target)
            refs = set()
            tasks, matters, evidence_hashes = set(), set(), set()
            non_assistant = False
            for ref in op.evidence:
                graded = by_episode.get(ref.episode_id)
                if graded is None:
                    raise ValueError("unknown evidence episode")
                ep = graded.episode
                observation = next((o for o in ep.observations if o.id == ref.observation_id), None)
                if (
                    observation is None
                    or observation.sha256 != ref.sha256
                    or digest(observation.content) != ref.sha256
                ):
                    raise ValueError("invalid evidence hash")
                key = (ref.episode_id, ref.observation_id)
                if key in refs:
                    raise ValueError("duplicate evidence reference")
                refs.add(key)
                if observation.kind != "assistant":
                    tasks.add(ep.task_id)
                    matters.add(graded.matter_id)
                    evidence_hashes.add(ref.sha256)
                    non_assistant = True
            if len(tasks) < self.policy.min_tasks or len(matters) < self.policy.min_matters:
                raise ValueError("insufficient independent support")
            if len(evidence_hashes) < 2 and max(self.policy.min_tasks, self.policy.min_matters) > 1:
                raise ValueError("repeated observation is not independent support")
            if not non_assistant:
                raise ValueError("assistant-only evidence")
            if op.action == "archive":
                cited = {(r.episode_id, r.observation_id) for r in op.evidence}
                if not any(
                    verdict.verdict == "fail"
                    and any((graded.episode.id, oid) in cited for oid in verdict.evidence_ids)
                    for graded in episodes
                    for verdict in graded.rubric
                ):
                    raise ValueError("archive requires cited failed rubric evidence")
            if op.candidate:
                # Candidate.evidence is a legacy observation map. Every cited observation must be
                # present in the batch; cross-episode provenance lives in EvidenceRef.
                counts = defaultdict(int)
                for ref in op.evidence:
                    counts[ref.observation_id] += 1
                bound = {
                    (
                        f"{ref.episode_id}:{ref.observation_id}"
                        if counts[ref.observation_id] > 1
                        else ref.observation_id
                    ): ref.sha256
                    for ref in op.evidence
                }
                qualified = {
                    f"{ref.episode_id}/{ref.observation_id}": ref.sha256 for ref in op.evidence
                }
                if op.candidate.evidence not in (bound, qualified):
                    raise ValueError("candidate evidence must bind every reference")
                supporting = [by_episode[ref.episode_id].episode for ref in op.evidence]
                if not set(op.candidate.required_tools) <= set.union(
                    *(set(e.tools) for e in supporting)
                ):
                    raise ValueError("unobserved tool requirement")
                if any(
                    not any(e.environment.get(k) == v for e in supporting)
                    for k, v in op.candidate.environment.items()
                ):
                    raise ValueError("unobserved environment")

    def _assess(self, namespace, operation, episodes, snapshot, decisions):
        if operation.candidate and any(
            r["identity"] == operation.candidate.identity and r["id"] not in operation.target_ids
            for r in snapshot
        ):
            return "pending", "exact_duplicate"
        cited = {(r.episode_id, r.observation_id) for r in operation.evidence}
        support = []
        for graded in episodes:
            ep_id = graded.episode.id
            if not any(pair[0] == ep_id for pair in cited):
                continue
            row = graded.model_dump(mode="json")
            row["episode"]["observations"] = [
                o for o in row["episode"]["observations"] if (ep_id, o["id"]) in cited
            ]
            row["rubric"] = [
                r for r in row["rubric"] if any((ep_id, oid) in cited for oid in r["evidence_ids"])
            ]
            support.append(row)
        state = {"operation": operation.model_dump(mode="json"), "support": support}
        if operation.action == "archive":
            state["prior_versions"] = [r for r in snapshot if r["id"] in operation.target_ids]
            questions = {
                "retirement_justified": (
                    "Do the cited graded failures justify retiring this exact prior memory version? "
                    "For a harm claim, verify applied_lessons shows this revision was used and the "
                    "advice caused or materially contributed to failure under its conditions. "
                    "Without applied use, require explicit evidence that a requirement in the "
                    "prior memory is obsolete. Mere co-occurrence or an unrelated failed criterion "
                    "is insufficient."
                )
            }
            scores = self._decision(namespace, state, questions)
            decisions.append({"phase": "retirement", "scores": scores})
            if scores is None:
                return "pending", "provider_error"
            if scores["retirement_justified"] <= self.policy.negative_threshold:
                return "rejected", "retirement_not_justified"
            if scores["retirement_justified"] < self.policy.threshold:
                return "pending", "uncertain_retirement"
            return "active", "accepted"
        questions = {
            "supported": "Is every proposed claim supported by the cited observations and rubric results?",
            "transferable": "Is this useful beyond the cited tasks under its stated conditions?",
            "actionable": "Does this prescribe a concrete action or decision?",
            "client_specific": "Is this advice specific to a client, matter, or deliverable rather than reusable practice?",
        }
        if operation.action in {"revise", "merge"}:
            state["prior_versions"] = [r for r in snapshot if r["id"] in operation.target_ids]
            questions["preserves_conditions"] = (
                "Does the proposed revision retain every still-supported condition and exception "
                "from its prior versions, unless the cited graded evidence justifies removing it?"
            )
        scores = self._decision(namespace, state, questions)
        decisions.append({"phase": "criteria", "scores": scores})
        if scores is None:
            return "pending", "provider_error"
        positive = ("supported", "transferable", "actionable") + (
            ("preserves_conditions",) if "preserves_conditions" in questions else ()
        )
        if (
            any(scores[k] <= self.policy.negative_threshold for k in positive)
            or scores["client_specific"] >= self.policy.threshold
        ):
            return "rejected", "failed_criteria"
        if (
            any(scores[k] < self.policy.threshold for k in positive)
            or scores["client_specific"] > self.policy.negative_threshold
        ):
            return "pending", "uncertain_criteria"
        # Every active memory is compared, in bounded requests, including targets so a revision
        # cannot silently erase a conflicting condition from its earlier version.
        for start in range(0, len(snapshot), self.policy.comparison_batch):
            batch = snapshot[start : start + self.policy.comparison_batch]
            questions = {
                f"contradiction_{i}": "Does the operation conflict with this memory under overlapping conditions?"
                for i in range(len(batch))
            }
            questions.update(
                {
                    f"duplicate_{i}": "Does the operation repeat this memory's advice and conditions?"
                    for i in range(len(batch))
                }
            )
            scores = self._decision(namespace, state | {"related": batch}, questions)
            decisions.append(
                {
                    "phase": "comparison",
                    "offset": start,
                    "related_ids": [r["id"] for r in batch],
                    "scores": scores,
                }
            )
            if scores is None:
                return "pending", "provider_error"
            if any(
                scores[f"contradiction_{i}"] > self.policy.negative_threshold
                for i in range(len(batch))
            ):
                return "pending", "conflict_or_uncertainty"
            for i, row in enumerate(batch):
                if (
                    row["id"] not in operation.target_ids
                    and scores[f"duplicate_{i}"] > self.policy.negative_threshold
                ):
                    return "pending", "duplicate_or_uncertainty"
        return "active", "accepted"

    def run(self, episodes: list[GradedEpisode], reviewer, cycle_id: str, *, _retry_id=None):
        episodes = [
            GradedEpisode.model_validate(
                g.model_dump(mode="json") if isinstance(g, GradedEpisode) else g
            )
            for g in episodes
        ]
        if not episodes or not cycle_id:
            raise ValueError("cycle requires graded episodes and an ID")
        if len({g.episode.id for g in episodes}) != len(episodes):
            raise ValueError("duplicate episode ID")
        namespace = episodes[0].episode.namespace
        if any(g.episode.namespace != namespace for g in episodes):
            raise ValueError("mixed namespaces")
        input_hash = digest([e.model_dump(mode="json") for e in episodes])
        prior = self.store.db.execute(
            "SELECT input_hash,result FROM consolidation_cycles WHERE namespace=? AND cycle_id=?",
            (namespace, cycle_id),
        ).fetchone()
        if prior:
            if prior["input_hash"] != input_hash:
                raise Conflict("cycle ID reused with different episodes")
            return json.loads(prior["result"])
        queued = self.store.db.execute(
            "SELECT payload FROM consolidation_queue WHERE namespace=? AND cycle_id=?",
            (namespace, cycle_id),
        ).fetchall()
        if any(digest(json.loads(row["payload"])["episodes"]) != input_hash for row in queued):
            raise Conflict("cycle ID reused with different episodes")
        for g in episodes:
            self.store.episode(g.episode)
        generation, snapshot = self.store.snapshot(namespace)
        try:
            started = time.perf_counter()
            proposed = reviewer.propose(episodes, snapshot)
            operations = [
                MemoryOperation.model_validate(
                    o.model_dump(mode="json") if isinstance(o, MemoryOperation) else o
                )
                for o in proposed
            ]
            self._validate(operations, episodes, snapshot)
            self.store.event(
                namespace,
                "consolidation_review",
                {
                    "cycle_id": cycle_id,
                    "status": "ok",
                    "model": getattr(reviewer, "model", "unknown"),
                    "usage": self._usage(reviewer),
                    "latency_ms": getattr(
                        reviewer, "last_latency_ms", (time.perf_counter() - started) * 1000
                    ),
                    "count": len(operations),
                    "snapshot_count": len(snapshot),
                },
            )
        except Conflict:
            raise
        except Exception as exc:
            if "proposed" not in locals():
                status = "pending" if isinstance(exc, ProviderError) else "rejected"
                reason = "provider_error" if status == "pending" else "reviewer_error"
            else:
                reason = (
                    VALIDATION_REASONS.get(str(exc), "invalid_proposal")
                    if isinstance(exc, ValueError)
                    else "invalid_proposal"
                )
                status = "pending" if reason == "insufficient_independent_support" else "rejected"
            review = [{"status": status, "reason": reason, "error_type": type(exc).__name__}]
            self.store.event(
                namespace,
                "consolidation_review",
                {
                    "cycle_id": cycle_id,
                    "status": "error",
                    "error": type(exc).__name__,
                    "model": getattr(reviewer, "model", "unknown"),
                    "usage": self._usage(reviewer),
                    "latency_ms": getattr(reviewer, "last_latency_ms", None),
                },
            )
            qid = self._enqueue(
                namespace,
                cycle_id,
                status,
                reason,
                episodes,
                operations=locals().get("operations"),
                review=review,
                retry_id=_retry_id,
            )
            return {
                "status": status,
                "reason": reason,
                "queue_id": qid,
                "error": type(exc).__name__,
                "operations": [],
            }
        comparison = list(snapshot)
        results = []
        review = []
        for op in operations:
            decisions = []
            verdict = self._assess(namespace, op, episodes, comparison, decisions)
            results.append(verdict)
            review.append(
                {
                    "operation_index": len(review),
                    "status": verdict[0],
                    "reason": verdict[1],
                    "decisions": decisions,
                }
            )
            if verdict[0] == "active" and op.candidate:
                comparison.append(
                    {
                        "id": op.target_ids[0] if op.target_ids else op.candidate.identity,
                        "revision": 1,
                        "status": "active",
                        "body": op.candidate.model_dump(mode="json"),
                        "identity": op.candidate.identity,
                    }
                )
        if any(status != "active" for status, _ in results):
            status, reason = next((s, r) for s, r in results if s != "active")
            qid = self._enqueue(
                namespace,
                cycle_id,
                status,
                reason,
                episodes,
                operations=operations,
                review=review,
                retry_id=_retry_id,
            )
            return {"status": status, "reason": reason, "queue_id": qid, "operations": []}
        if self.shadow:
            return {
                "status": "shadow",
                "operations": [o.model_dump(mode="json") for o in operations],
            }
        with self.store.transaction():
            if self.store.generation(namespace) != generation:
                raise Conflict("memory changed during assessment")
            latest = {r["id"]: r for r in self.store.lessons(namespace, active=False)}
            for op in operations:
                for target in op.target_ids:
                    if latest[target]["revision"] != op.expected_revisions[target]:
                        raise Conflict("revision mismatch")
            committed = []
            for op in operations:
                target = op.target_ids[0] if op.target_ids else None
                identity = op.candidate.identity if op.candidate else latest[target]["identity"]
                memory_id = target or identity
                prior = latest.get(memory_id)
                revision = prior["revision"] + 1 if prior else 1
                status = "archived" if op.action == "archive" else "active"
                body = (
                    (
                        op.candidate.model_dump(mode="json")
                        | {"kind": op.kind, "work_kind": op.work_kind}
                    )
                    if op.candidate
                    else prior["body"]
                )
                self.store.db.execute(
                    "INSERT INTO revisions(namespace,id,revision,status,body,identity,validation) VALUES (?,?,?,?,?,?,?)",
                    (namespace, memory_id, revision, status, canonical(body), identity, "batch"),
                )
                self.store.db.execute(
                    "DELETE FROM search WHERE namespace=? AND id=?", (namespace, memory_id)
                )
                if status == "active":
                    c = op.candidate
                    self.store.db.execute(
                        "INSERT INTO search VALUES (?,?,?,?)",
                        (
                            namespace,
                            memory_id,
                            revision,
                            " ".join([c.trigger, c.lesson, *c.preconditions, *c.exceptions]),
                        ),
                    )
                for ref in op.evidence:
                    self.store.db.execute(
                        "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)",
                        (namespace, memory_id, ref.episode_id, ref.observation_id, ref.sha256),
                    )
                if op.action == "merge":
                    for merged_id in op.target_ids[1:]:
                        old = latest[merged_id]
                        self.store.db.execute(
                            "INSERT OR IGNORE INTO evidence(namespace,lesson_id,episode_id,observation_id,hash) "
                            "SELECT namespace,?,episode_id,observation_id,hash FROM evidence "
                            "WHERE namespace=? AND lesson_id=?",
                            (memory_id, namespace, merged_id),
                        )
                        self.store.db.execute(
                            "INSERT INTO revisions(namespace,id,revision,status,body,identity,validation) VALUES (?,?,?,?,?,?,?)",
                            (
                                namespace,
                                merged_id,
                                old["revision"] + 1,
                                "archived",
                                canonical(old["body"]),
                                old["identity"],
                                "batch",
                            ),
                        )
                        self.store.db.execute(
                            "DELETE FROM search WHERE namespace=? AND id=?", (namespace, merged_id)
                        )
                committed.append({"id": memory_id, "revision": revision, "action": op.action})
            if operations:
                self.store._bump(namespace)
            self.store.db.execute(
                "UPDATE consolidation_queue SET status='resolved',reason='accepted' "
                "WHERE namespace=? AND cycle_id=?",
                (namespace, cycle_id),
            )
            result = {"status": "active", "operations": committed}
            self.store.db.execute(
                "INSERT INTO consolidation_cycles VALUES (?,?,?,?)",
                (namespace, cycle_id, input_hash, canonical(result)),
            )
        self.store.event(
            namespace,
            "consolidation_result",
            {"cycle_id": cycle_id, "status": "active", "operations": committed},
        )
        return result

    @staticmethod
    def _usage(reviewer):
        usage = getattr(reviewer, "last_usage", None)
        return usage.model_dump(mode="json") if hasattr(usage, "model_dump") else usage
