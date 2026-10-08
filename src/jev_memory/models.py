"""Validated boundary objects and deterministic content identities."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, max_length=16000)]
Identifier = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[\w./:-]+$")]


def canonical(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Observation(Record):
    id: Identifier
    kind: Literal["user", "tool", "verifier", "assistant"]
    content: Text
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def check_hash(self):
        if self.sha256 != digest(self.content):
            raise ValueError("observation hash mismatch")
        return self

    @classmethod
    def create(cls, id: str, kind: str, content: str):
        return cls(id=id, kind=kind, content=content, sha256=digest(content))


class Episode(Record):
    id: Identifier
    task_id: Identifier
    task_group: Identifier
    namespace: Identifier
    task: Text
    host_revision: Text
    solver_model: Text
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)
    tools: list[Identifier] = Field(default_factory=list, max_length=64)
    observations: list[Observation] = Field(min_length=1, max_length=128)
    solver_claimed_success: bool | None = None
    verifier_result: Literal["pass", "fail", "unknown"] = "unknown"
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_observations(self):
        if len({o.id for o in self.observations}) != len(self.observations):
            raise ValueError("duplicate observation IDs")
        if self.verifier_result != "unknown" and not any(
            o.kind == "verifier" for o in self.observations
        ):
            raise ValueError("verifier result requires a verifier observation")
        return self

    @property
    def trace_hash(self):
        return digest(self)


class Candidate(Record):
    namespace: Identifier
    trigger: Text
    lesson: Text
    preconditions: list[Text] = Field(min_length=1, max_length=16)
    exceptions: list[Text] = Field(default_factory=list, max_length=16)
    evidence: dict[Identifier, str] = Field(min_length=1, max_length=128)
    required_tools: list[Identifier] = Field(default_factory=list, max_length=64)
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)

    @property
    def identity(self):
        return digest(self.model_dump(exclude={"evidence"}))


class Draft(Record):
    """Writer-owned prose only. Namespace and evidence hashes are host-owned."""

    trigger: Text
    lesson: Text
    preconditions: list[Text] = Field(min_length=1, max_length=16)
    exceptions: list[Text] = Field(default_factory=list, max_length=16)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=128)
    required_tools: list[Identifier] = Field(default_factory=list, max_length=64)
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)

    def resolve(self, episode: Episode) -> Candidate:
        observations = {o.id: o for o in episode.observations}
        if any(id not in observations for id in self.evidence_ids):
            raise ValueError("unknown evidence ID")
        return Candidate(
            namespace=episode.namespace,
            evidence={id: observations[id].sha256 for id in self.evidence_ids},
            **self.model_dump(exclude={"evidence_ids"}),
        )


class Task(Record):
    id: Identifier
    namespace: Identifier
    text: Text
    tools: list[Identifier] = Field(default_factory=list, max_length=64)
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)


class Policy(Record):
    threshold: float = Field(default=0.8, gt=0.5, le=1, allow_inf_nan=False)
    admission_threshold: float | None = Field(default=None, gt=0.5, le=1, allow_inf_nan=False)
    retrieval_threshold: float | None = Field(default=None, gt=0.5, le=1, allow_inf_nan=False)
    negative_threshold: float = Field(default=0.2, ge=0, lt=0.5, allow_inf_nan=False)
    shortlist: int = Field(default=20, ge=1, le=100)
    max_lessons: int = Field(default=3, ge=1, le=10)
    token_budget: int = Field(default=1200, ge=1, le=16000)
    related_limit: int = Field(default=100, ge=1, le=200)


class Usage(Record):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Decision(Record):
    scores: dict[str, float]
    model: str
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def finite_scores(self):
        if any(not 0 <= x <= 1 for x in self.scores.values()):
            raise ValueError("scores must be finite numbers in [0,1]")
        return self
