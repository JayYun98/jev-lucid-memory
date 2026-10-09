"""Explicit chat-model adapter for proposed memory changes, never direct writes."""

from __future__ import annotations

import json
import time
from typing import Literal

import httpx
from pydantic import Field, TypeAdapter

from .models import Identifier, Record, Text, Usage, canonical
from .providers import PREAMBLE, LocalWriter, OpenRouterWriter, ProviderError, usage


class Reference(Record):
    episode_id: Identifier
    observation_id: Identifier


class LessonText(Record):
    trigger: Text
    lesson: Text
    preconditions: list[Text] = Field(min_length=1)
    exceptions: list[Text] = Field(default_factory=list)
    required_tools: list[Identifier] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)


class DraftOperation(Record):
    action: Literal["add", "revise", "merge", "archive"]
    kind: Literal["checklist", "practice"] = "practice"
    work_kind: Literal["analysis", "drafting", "review", "general"] = "general"
    candidate: LessonText | None = None
    evidence: list[Reference] = Field(min_length=1)
    target_ids: list[Identifier] = Field(default_factory=list)
    expected_revisions: dict[Identifier, int] = Field(default_factory=dict)

    def resolve(self, episodes):
        from .consolidation import EvidenceRef, MemoryOperation
        from .models import Candidate

        by_id = {e.episode.id: e.episode for e in episodes}
        refs = []
        for ref in self.evidence:
            episode = by_id[ref.episode_id]
            observation = next(o for o in episode.observations if o.id == ref.observation_id)
            refs.append(EvidenceRef(**ref.model_dump(), sha256=observation.sha256))
        hashes = {f"{ref.episode_id}/{ref.observation_id}": ref.sha256 for ref in refs}
        candidate = (
            None
            if self.candidate is None
            else Candidate(
                namespace=episodes[0].episode.namespace,
                kind=self.kind,
                work_kind=self.work_kind,
                evidence=hashes,
                **self.candidate.model_dump(),
            )
        )
        return MemoryOperation(
            candidate=candidate, evidence=refs, **self.model_dump(exclude={"candidate", "evidence"})
        )


class BatchWriter:
    """Reuse a configured writer's recipient, authentication and transport.

    The writer sees support grades only. Queries/held-out grades belong to the
    cycle runner and must never be passed to this method.
    """

    def __init__(self, writer: LocalWriter | OpenRouterWriter):
        self.writer = writer
        self.model = writer.model
        self.last_usage = None
        self.last_latency_ms = 0.0

    def propose(self, episodes, snapshot):
        instructions = (
            PREAMBLE + "You review graded training episodes and an existing memory bank. "
            'Return only JSON {"operations": [...]} with zero to four operations. '
            "Propose reusable checklist items or practice notes, not client facts. "
            "Keep necessary conditions and exceptions explicit. Do not invent a successful "
            "fix from a failed run. Require evidence from independent tasks and matters. "
            "Compare with existing lessons: add genuinely new advice, revise an existing "
            "lesson to preserve supported conditions/exceptions, merge overlapping lessons, "
            "or archive advice whose harm is supported by graded evidence. "
            "Never broaden a lesson merely to reduce duplicates. No names, prices, secrets, "
            "or one-task details in the lesson. Copy episode and observation IDs exactly. The host assigns evidence hashes. "
            "Use the current target revisions, never invent them. An empty list is valid. IMPORTANT: memory_snapshot contains stored lessons, support_episodes contains runs. Only IDs present in memory_snapshot can be target_ids. When memory_snapshot is empty, only ADD is possible: target_ids=[], expected_revisions={}, candidate must be an object with trigger, lesson, preconditions, exceptions. MERGE merges existing lessons, never episodes. REVISE needs exactly one stored lesson target; MERGE needs two or more; ARCHIVE needs one and candidate=null. ADD/REVISE/MERGE always require candidate prose. "
            "Operation JSON schema: " + canonical(DraftOperation.model_json_schema())
        )
        state = {
            "support_episodes": [e.model_dump(mode="json") for e in episodes],
            "memory_snapshot": snapshot,
        }
        start = time.perf_counter()
        self.last_usage = None
        try:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": canonical(state)},
            ]
            totals = Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)
            with httpx.Client(
                timeout=self.writer.timeout, transport=self.writer.transport
            ) as client:
                for attempt in range(2):
                    response = client.post(
                        self.writer.endpoint,
                        headers={"Authorization": f"Bearer {self.writer._key}"},
                        json={
                            "model": self.model,
                            "temperature": 0,
                            "max_tokens": 4000,
                            "chat_template_kwargs": {"enable_thinking": False},
                            "response_format": {"type": "json_object"},
                            "messages": messages,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    spent = usage(data.get("usage", {}))
                    totals = Usage(
                        input_tokens=totals.input_tokens + spent.input_tokens,
                        output_tokens=totals.output_tokens + spent.output_tokens,
                        cost_usd=(
                            totals.cost_usd + spent.cost_usd
                            if totals.cost_usd is not None and spent.cost_usd is not None
                            else None
                        ),
                    )
                    self.last_usage = totals
                    content = data["choices"][0]["message"]["content"]
                    try:
                        body = json.loads(content)
                        if set(body) != {"operations"} or not isinstance(body["operations"], list):
                            raise ValueError("invalid operation envelope")
                        if len(body["operations"]) > 4:
                            raise ValueError("too many operations")
                        drafts = TypeAdapter(list[DraftOperation]).validate_python(
                            body["operations"]
                        )
                        return [draft.resolve(episodes) for draft in drafts]
                    except (ValueError, KeyError, StopIteration) as exc:
                        if attempt:
                            raise
                        # Only schema repair, never weaken gates or change the recipient.
                        messages += [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": "Invalid proposal ("
                                + type(exc).__name__
                                + "). Correct the JSON schema and reference IDs. ADD requires candidate prose "
                                "and no targets; REVISE/MERGE targets must be existing memory IDs; "
                                "ARCHIVE alone has candidate=null. Never target episode IDs. "
                                "If no supported change is possible, return an empty operations list.",
                            },
                        ]
        except Exception as exc:
            raise ProviderError(type(exc).__name__) from None
        finally:
            self.last_latency_ms = (time.perf_counter() - start) * 1000
