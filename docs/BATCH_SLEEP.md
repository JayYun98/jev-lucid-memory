# Batch Sleep and repeated evaluation

The batch path extends the single-episode CLI with graded support episodes and a
review of the existing memory bank. It implements the mechanics of a repeated
Wake–Sleep loop; it does not reproduce Harvey's legal benchmark results.

## Flow

1. A host solves support tasks, recording tool observations and deliverables.
2. Separate graders assess rubric criteria. The task solver does not receive grades.
3. A chat-model reviewer sees the graded support episodes and current memory.
4. It proposes **add, revise, merge or archive** operations.
5. Code binds evidence IDs to stored hashes and checks independent support. The
   decision model assesses support, usefulness, identifying details and conflicts.
6. An accepted batch commits atomically against its original memory generation.
   Failed or uncertain proposals remain inspectable in a durable queue.
7. The next Wake uses the revised memory. Held-out tasks never teach Sleep.

A lesson has a `kind` (`checklist` or `practice`) and `work_kind` (`analysis`,
`drafting`, `review`, `general`), alongside its conditions, exceptions and evidence.
Jev, local Clef and the local generative decision baseline share the gate interface.
The writing model remains a separate choice.

## Run a batch

Start a local writer and [Clef server](LOCAL.md). Supply JSONL containing one
`GradedEpisode` per line; the schema is in
[consolidation.py](../src/jev_memory/consolidation.py). Each record wraps an existing
`Episode` with `matter_id`, criterion verdicts, applied lesson revisions and deliverable references.
The supplied [example](../examples/graded.jsonl) is an authored synthetic fixture,
not a real agent execution or evidence of generalization. Use real,
trusted task and matter IDs; assigning different IDs to copies is not new evidence.

```sh
# Inspect a proposal in shadow mode.
uv run jev-memory --db local/memory.db batch-sleep examples/graded.jsonl \
  --cycle-id cycle-1 --writer-model "$MEMORY_MODEL"

# Assess again and apply an accepted batch.
uv run jev-memory --db local/memory.db --apply batch-sleep examples/graded.jsonl \
  --cycle-id cycle-1 --writer-model "$MEMORY_MODEL"

# Inspect the queue; no model is contacted.
uv run jev-memory --db local/memory.db queue --namespace batch-demo

# Generate and revalidate a corrected proposal; this is not an approval bypass.
uv run jev-memory --db local/memory.db --apply retry 1 \
  --namespace batch-demo --writer-model "$MEMORY_MODEL"
```

`--provider local-llm --model "$MEMORY_MODEL"` selects a local generative gate.
`--provider openrouter` or `--provider typesafe` explicitly selects hosted Jev.
The reviewer stays local unless `--cloud-writer` is supplied.

**Shadow mode still persists source episodes and audit traces locally.** It does not
activate memory or enqueue live changes. Use a private database location and redact
sensitive material before supplying it. The CLI package name remains `jev-memory`.

## Policy and consistency

The batch defaults require at least two distinct tasks and two distinct matters,
with non-assistant evidence. Repeated observation hashes cannot alone establish
independence. This is a provenance policy, not statistical proof of independence.

Pass a JSON `--consolidation-policy` to set `min_tasks`, `min_matters`, `threshold`,
`negative_threshold` and `comparison_batch`. Policy calibration belongs on development
data, never on held-out test grades.

All active lessons are compared in bounded batches. There is no 100-lesson admission
cutoff in this path. A provider context limit still fails closed; bounded comparison
count is not an unlimited-token guarantee. The reviewer currently receives the whole
memory bank, so very large banks will need a separately verified hierarchical review.

Revision targets must match current active revisions. Merge archives its source
records while preserving history. Archive is backed by cited negative rubric evidence.
The gate checks whether revised conditions and exceptions remain justified. These
model judgments can be wrong; hashes and transactions do not establish lesson quality.

## Repeated local experiment

```sh
uv run python scripts/eval_cycles.py --model "$MEMORY_MODEL" \
  --decision clef --cycles 2 --run-id clef-smoke --output local/clef-cycles

# The runner supports 1–10 cycles. Use a new run ID and output path for changed settings.
uv run python scripts/eval_cycles.py --model "$MEMORY_MODEL" \
  --decision llm --cycles 10 --run-id llm-ten --output local/llm-ten
```

The script makes actual local model calls for task plans, two isolated grader
requests and proposed memory changes. The two grader requests use the same local
model with different instructions; they are not statistically independent models.
A deterministic interpreter separately checks output and side effects. Generated
Python or shell code is never executed.

The small synthetic suite is a mechanics test: pagination support tasks, familiar
pagination queries and an unseen sorting task family. It is deliberately easy and
cannot establish legal reasoning quality or a general learning advantage.

The runner compares no memory, full eligible memory and selected eligible memory
from the same frozen bank. Full-memory budget overflow is explicit; it does not
silently truncate that arm. Later support tasks use the preceding memory. Query
answers and grades are excluded from future Sleep. Task and matter identities cannot
cross the support/query split. Manifests record snapshots, grading disagreement,
per-cohort outcomes, tokens, latency and tool calls. API charges are zero for local
inference; electricity and hardware costs are not estimated.

Resume uses the same command and verifies the manifest and bank. If interrupted after
an unreceipted mutating Sleep, inspect the run rather than automatically replaying it.
The reusable runner and callable host/grader adapters live in
[cycles.py](../src/jev_memory/cycles.py).

## Verification — 2026-10-09

The integrated suite passed **82 tests** with **93% statement coverage**. Ruff lint
and wheel/sdist builds passed. Regression checks cover atomic rollback, stale
revisions, legacy archived identities, cross-namespace isolation, independent
support, comparison beyond 100 lessons, grader disagreement and frozen query banks.
These tests validate the implementation contracts, not the quality of a model judgment.

Actual local inference used Qwen3.8-27B-4bit as writer, solver and the generative
baseline, alongside the official Clef-Flash BF16 decision backend on MPS.

| Decision backend | Cycles | Support tasks | Query tasks per arm | Stored lessons | No / full / selected passes |
|---|---:|---:|---:|---:|---|
| Clef-Flash | 1 | 2 | 3 | 0 | 3/3 · 3/3 · 3/3 |
| Local Qwen | 2 | 4 | 6 | 1 | 6/6 · 6/6 · 6/6 |

Qwen's generated lesson was committed and selected on two of six queries. The
second cycle retained the bank and proposed no new operation. Clef rejected its
proposal; its three comparison arms therefore all had empty memory. Thresholds
were not relaxed to force admission. These are separate smoke runs, not a matched
backend quality comparison. All no-memory tasks already passed: **no performance
improvement was established**. External inference charges were **$0**. A separate actual `batch-sleep --apply`
CLI call also committed one lesson from the observed support episodes.
See the [execution receipt](results/batch-cycles.json) for revisions and diagnostics.

## Scope

This release adds batch consolidation, explicit cyclic execution and a local synthetic
harness. It does not automatically export arbitrary Hermes conversations, schedule
background jobs, or include Harvey's LAB documents and expert rubric dataset.
Cloud Jev quality and legal-task transfer remain separate evaluations.
