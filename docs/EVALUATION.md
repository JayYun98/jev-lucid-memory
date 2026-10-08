# Evaluation

## Executed locally — 2026-10-09

Apple M5 Pro, 64 GB unified memory, macOS; Python 3.12.8.
Clef used its official BF16 reference implementation on PyTorch MPS:
`Cloudflare/clef-flash@fde727a287004204b7518dcc983fe64379776712`.
Runtime: torch 2.14.1, torchvision 0.29.1, transformers 5.19.0.
The local writer/solver/generative gate was `qwen3.6-35b-a3b-mxfp4`, served by
`mlx_lm.server`. Both models were loaded locally. External inference charges: **$0**.
Hardware/electricity cost was not estimated. Concurrent diagnostic requests and
co-resident models make these timings unsuitable as isolated throughput claims.

### Engineering checks

- 52 offline tests passed; 94% statement coverage of `jev_memory`.
- Ruff checks, wheel/sdist build and clean Python 3.13 wheel installation/CLI smoke passed.
- Real Hermes manifest/config/hook dispatcher plus local inference passed in an isolated profile.
  A second sender received no first-sender memory. No live profile was changed.
- Clef `/health` and real `systemone` requests succeeded. Oversized input returned
  HTTP 413 without truncation; an unknown model returned HTTP 422.
- Jev/OpenRouter and direct TypeSafe SDK paths were contract-tested, **not called live**.
- CUDA, CPU inference, Windows, complete Hermes agent sessions and cloud Jev quality
  are not validated by these results.

### Shared-candidate applicability replay

Same authored pagination lesson, six fixed positive/counterexample queries, threshold 0.8.
This explicitly bypasses storage selection for replay only; production admission is unchanged.

| Decision model | Applicability labels correct | Full admission → Wake selection |
|---|---:|---:|
| Local Qwen | 6/6 | 6/6 |
| Local Clef | 6/6 | 3/6 |

Clef admission scored supported=0.7708, transferable=0.9367, actionable=0.6298.
It therefore left the lesson pending and selected nothing in the full path. Its direct
applicability scores were 0.9320/0.8808/0.9060 for positive queries and
0.0995/0.2436/0.1394 for counterexamples. **Good retrieval classification did not
compensate for overly restrictive admission.** Thresholds were not tuned to these results.

The same local reflector also generated a candidate from the support episode;
Qwen admitted it and Clef left it pending. This checks the real generation → admission
path; it does not prove that generated lessons improve future tasks.

Raw synthetic receipts: [Clef](results/clef-replay.json), [Qwen](results/qwen-replay.json).
The `retrieval_only` column in that script uses the gate's admitted store and is diagnostic,
not the independent A1 baseline below.

### Actual solver plans and deterministic execution

Six small tasks across pagination, transient reads versus non-idempotent writes,
and version-dependent ordering. A real local solver emits a bounded action plan.
A restricted interpreter executes it and checks exact output plus side effects.
There is no arbitrary generated-code execution and no model judge.

| Arm | Passed | Gain vs no memory | Harm vs no memory |
|---|---:|---:|---:|
| A0 — no memory | 3/6 | — | — |
| A1 — lexical retrieval | 5/6 | 2 | 0 |
| A2 — local Qwen gate | 3/6 | 0 | 0 |
| A3 — local Clef gate | 3/6 | 0 | 0 |

[Full rollout receipt](results/rollouts.json). **No gated-memory improvement was
established.** Clef held all three support lessons pending. Qwen admitted them and
selected relevant lessons, but the solver still failed some plans. Selection is not utility.

All arms share authored support fixtures, schema, shortlist size, memory count and
1,200-byte conservative context budget. The mixed memory store is frozen during queries;
generation checks enforce this. Solver, temperature, output limit and query contracts are
fixed. The verifier and expected outputs are excluded from solver prompts. Task contracts
are deliberately simple and explicit; this is a diagnostic suite, not a hard benchmark.

One repeat is reported. Three families and six queries cannot establish statistical
significance, calibration or broad generalization. Development checks are not a held-out
external benchmark. No SkillLearnBench, AppWorld, LILO or learned-code-skill result is claimed.

## Reproduce

Start both local servers; see [LOCAL.md](LOCAL.md). No API keys are required.

```bash
uv run python scripts/eval_local.py --provider clef --model clef-flash \
  --writer-model "$MEMORY_MODEL" --output local/clef.json
uv run python scripts/eval_local.py --provider llm --model "$MEMORY_MODEL" \
  --writer-model "$MEMORY_MODEL" --output local/llm.json
uv run python scripts/eval_rollouts.py --solver-model "$MEMORY_MODEL" \
  --repeats 1 --output local/rollouts.json
```

Model revision, policy/instruction/state hashes, usage and timing are recorded.
Support in the rollout suite is authored, not an autonomous solver learning episode.
The separate Sleep smoke exercises the real reflector. Do not merge those claims.

## Calibration and next external evaluation

`--policy policy.json` supports separate `admission_threshold` and `retrieval_threshold`
(omitted values inherit `threshold`). The defaults remain 0.8 with negative threshold 0.2.
Model Noul scores share an interface, not necessarily calibration. Choose thresholds
on development families, freeze them, and only then evaluate unseen families. A threshold
change is not evidence of improved performance without a new, independent evaluation.

The next external pilot should use [SkillLearnBench](https://github.com/cxcscmu/SkillLearnBench)
support/query instances, with no query descriptions supplied to reflection, followed by
[AppWorld](https://github.com/StonyBrookNLP/appworld) under its official split rules.
Those environments have additional Docker/data/provider requirements; they were inspected
but not installed or run in this release. Any changed protocol must be labeled an adaptation.

Keep A0/A1/A2/A3 stores separate. Generate one shared candidate corpus for gate-only replay;
report closed-loop reflection separately. Freeze memory for offline evaluation; online
updates are a different experiment. Record gain/harm, task success, gate errors, memory
reuse, total tokens, wall time and hardware time. Group uncertainty by task family.
