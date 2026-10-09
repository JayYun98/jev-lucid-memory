# Implementation record

## Scope

v0.1 implements the portable v1 memory loop from the supplied wake–sleep design:
three independent gates, an immutable evidence-backed SQLite store, standalone JSONL Sleep,
Wake with scope filters and abstention, a disabled-by-default Hermes plugin, and local evals.
Executable code skills, Stitch abstraction learning, and Jev weight training are **not implemented**.

## Decisions

- Default inference is loopback-only Clef. Cloud Jev requires explicit provider selection.
- Text generation is a separate local OpenAI-compatible endpoint. No paid evaluation requests.
- Shadow records traces and decisions, but never activates lessons or injects context.
- Source observations are caller-supplied; hashes detect changed bytes, not factual truth.
- Unknown/error/contradiction is pending. Pending decisions can be retried.
- Admission assesses the full active namespace up to a bounded limit; excess remains pending.
- Each lesson revision is immutable. Expected revisions and namespace generations protect writes.
- Retries of the same episode/candidate do not append revisions or evidence rows.
  Distinct episode IDs remain distinct provenance; this is not a statistical independence claim.
- Semantic revisions must pass admission again. Lifecycle changes can only retire records.
- Task namespace, required tools and exact environment values are filtered before applicability.
- FTS5/BM25 shortlists are lexical and can miss paraphrases. This is a known recall limit.
- Token budget is a conservative UTF-8 byte upper bound, not a tokenizer-specific measurement.

## Upstream inspection

- Hermes local checkout: `d84ece48b8552501660be229797e2d2aa4cee8db`.
- `pre_llm_call` accepts `user_message`, `session_id`, `sender_id`, `turn_id`; returns `context`.
- `request_background_review` PR 128885 was open/unmerged on 2026-10-09.
- Built-in reviewer writes are not intercepted. Use an isolated profile and explicitly disable
  `auxiliary.background_review.enabled` when comparing automatic memory systems.
- No modifications to the user's live Hermes profile or memory.

## Influences

[Jev Wiki](https://github.com/JayYun98/jev-wiki) is the author's earlier, now public project:
Jev decides, a host LLM writes, code hashes and commits evidence-backed Markdown.
This implementation carries over the *division of responsibility*, revision checks, explicit
uncertainty and no silent truncation. It does not import Jev Wiki or copy its sources.
The reference is provenance, not a required dependency or an implemented integration.
