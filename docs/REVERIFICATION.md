# Concept scope and re-verification — 2026-10-09

Starting revision: `585721e63bef28a14031476879b78e0c2251cbcb`.

## What was actually referenced

The original implementation used the supplied design document and inspected Hermes,
its Jev routing/trigger plugins, Clef's reference interface, and Jev Wiki's division
of responsibilities. It did **not** integrate or execute DreamCoder, LILO or Stitch.
Their role in the original scope was conceptual/future work.

This follow-up inspected the official repositories and DreamCoder's main loop:

- [DreamCoder](https://github.com/ellisk42/ec/blob/master/dreamcoder/dreamcoder.py)
  contains program search, grammar induction and recognition-model training, including
  generated Helmholtz tasks. Our `sleep()` only reflects on supplied observations.
- [LILO](https://github.com/gabegrand/lilo) combines program synthesis with library
  induction and language descriptions. Our stored objects are prose lessons, not programs.
- [Stitch](https://github.com/mlb2251/stitch) discovers abstractions and rewrites programs.
  Our duplicate/contradiction checks do not implement that operation.

No upstream training or benchmark was run. A repository specifically named “dreaming”
has not been identified; these comparisons refer to the repositories in the supplied brief.

## Exact implementation claim

An executable **proof of concept for conditional lesson memory**, not a production
service and not a reproduction of DreamCoder's learning algorithm.

Implemented: local model decisions, reflection, evidence checks, durable storage,
retrieval gating, namespace/environment filtering, revisions and an opt-in host hook.
Not implemented: dream-task generation, executable learned skills, library compression,
recognition-network training or demonstrated general task-success improvement.

The earlier six-task rollout used authored support traces and a one-shot action plan
executed by a deterministic interpreter. It did not evaluate autonomous support collection
or a multi-turn agent learning loop. Real inference alone does not establish learning.

## Reproduced defect and correction

At the starting revision, requesting Sleep against an unavailable loopback model returned
`{"reflected":false,"results":[]}` with exit status 0. Retrieval transport errors and
nested admission failures could also appear as successful CLI runs.

The correction preserves fail-closed memory behavior while reporting `provider_error`
and a nonzero CLI exit status. A valid model decision to abstain remains successful.
Three targeted regression tests failed before the fix and passed after it; the full suite
now passes **55 tests**. The real unavailable-port CLI reproduction now exits with status 1.

## Fresh real inference

The old Qwen 35B local path was no longer present. No replacement weights were downloaded.
Used the cached `mlx-community/Qwen3-4B-8bit` revision
`0348ad770d2ae658ca47b0579b2d2c37b20bbcac` for generation/admission and Clef's existing
pinned reference server for additional retrieval judgments. This is **not** a repeat
of the original 35B benchmark.

Observed with a newly generated lesson and a temporary SQLite database:

- Local reflection generated a lesson and Qwen admitted it.
- Closing/reopening the database preserved it; the positive task retrieved it.
- Another namespace retrieved nothing.
- Archiving, reopening and retrieving returned no active lesson.
- The writer produced a generic cursor-pagination suggestion without the intended
  “all records requested” condition. Both Qwen and Clef selected it for the first-page
  query too. The intended conditional-learning expectation was therefore **not met**.

This does not prove that the solver would fetch unwanted pages: the generated lesson
was broader than the authored lesson used in the original replay. It shows why correct
selection on a hand-authored candidate does not validate the full learning loop.
No downstream solver utility claim is made for this run.

[Raw synthetic receipt](results/reverification.json). Reproduce with local servers:

```bash
uv run python scripts/reverify_lifecycle.py --model "$MEMORY_MODEL" \
  --output local/reverification.json
```

Model quality failures are recorded alongside successful storage mechanics. A nonempty
memory store, valid JSON and passing unit tests are not proof of useful learning.
