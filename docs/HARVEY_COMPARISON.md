# Harvey Wake–Sleep: implementation gap review

Reviewed on 2026-10-09 against code revision `04c4735` and the user-supplied full
text of **The Return of Wake-Sleep**, linked to
[Niko Grupen's post](https://x.com/nikogrupen/status/2108226990792900876).
The supplied text is the source for the article's protocol and reported numbers;
this review does not independently reproduce its experiments or inspect unpublished code.

## Conclusion

Jev Lucid Memory implements components of the same idea: a writer proposes lessons,
a decision model gates admission, and another decision checks applicability at Wake.
It does **not** implement the complete Harvey training-and-evaluation loop.
The most significant missing mechanism is batch memory revision driven by graded
rollouts across repeated cycles—not an overnight scheduler.

The previous comparison used the separate Anthropic Managed Agents announcement
because the X article could not be fetched. That comparison did not establish the
contents of this article. Harvey's own Jev-gated experiment is the reference here.

## Protocol comparison

| Article stage | Current implementation | Gap |
|---|---|---|
| Agent completes long legal tasks and produces deliverables | Core accepts caller-supplied episodes; Hermes provides a Wake hook | No end-to-end legal rollout runner, deliverable collector or automatic trace export |
| Two LLM judges score expert rubric criteria; grades hidden from solver | `Episode` has pass/fail/unknown plus generic verifier observations | No structured criterion verdicts, dual-grader harness or enforced role separation for grades |
| Review model reads graded trajectory and deliverables | `writer.propose(episode)` drafts up to three lessons from one episode | No complete graded-deliverable pipeline; no checklist/practice-note types |
| Merge, revise, prune existing memory; assess prior usefulness | Admission detects duplicates/conflicts; storage can merge evidence, replace revisions and archive | No reviewer that proposes semantic add/revise/merge/drop changes against the memory bank |
| Jev rejects client detail, narrow overfitting and single-task support | Generic supported/transferable/actionable questions and evidence checks | No explicit client-detail gate or independently counted cross-task support requirement |
| Store by analysis/drafting/review/general | Namespace-scoped structured lessons in SQLite | No work-kind taxonomy; namespaces are isolation boundaries, not substitutes for categories |
| Repeat ten Wake–Sleep cycles | Separate Sleep and Wake calls | No cycle coordinator, frozen cycle snapshots or lesson-utility feedback |
| Retrieve helpful lessons with Jev | FTS5 shortlist, applicability gate, context limits | Similar intent, but not equivalent search coverage; no full-memory comparison arm |

The article's use of Wake includes the complete agent rollout. Our `wake()` names
only retrieval and advice construction; the host performs the rest.

## Material findings

### 1. Evidence deduplication is not semantic consolidation

[Memory.sleep](../src/jev_memory/core.py) calls `writer.propose(episode)` without the
existing memory bank, then admits each candidate. In [Store.save](../src/jev_memory/store.py),
a duplicate preserves the existing revision and adds evidence rows. It does not
rewrite the lesson, merge its conditions or learn new exceptions.

The `replace_id` and expected-revision path provides useful storage infrastructure.
It is not an automated revision policy. Likewise, manual archiving does not assess
whether a lesson helped or harmed later tasks.

**Needed:** a reviewer receives graded support episodes and a versioned memory
snapshot, proposes explicit add/revise/merge/archive operations, and submits those
operations to the decision gate. Each operation must preserve provenance and be
committed against the expected snapshot. Reviewers must see counterexamples before
broadening or narrowing conditions.

### 2. The overfitting policy is weaker than the article's

A single episode can currently support an active lesson. Observation hashes verify
identity, not independent support. Duplicate evidence accumulated later does not
make cross-task support a prerequisite for initial admission. The generic
transferability question is not an explicit check for client names, amounts or
matter-specific facts. The writer's "keep scope narrow" instruction also needs a
clear distinction between useful preconditions and one-instance overfitting.

**Needed:** distinguish observation, episode, task and matter identities; count
independent supporting tasks; assess identifying detail and generalizability; keep
single-task candidates pending under a Harvey-style policy. Do not infer a task-count
threshold beyond what is stated in the article, and do not replace legitimate scope
conditions with vague universally applicable advice.

### 3. Current limits do not match the reported memory scale

[Policy](../src/jev_memory/models.py) defaults to `related_limit=100`. Admission
checks all active lessons in the namespace and returns pending when their count
exceeds this limit. With unique sequential additions, 101 active lessons can be
reached; subsequent ordinary admissions are then deferred. The article reports
122 lessons in its first cycle, already beyond this default path.

Increasing the number alone is not a scalable fix. Each candidate currently adds
a duplicate and contradiction question for every active lesson. A larger context
can exceed provider budgets and obscure relevant conflicts.

**Needed:** bounded related-lesson retrieval plus explicit coverage/overflow handling,
then batch consolidation and a memory-size budget. Test at and above 122 lessons;
uninspected relevant conflicts must not silently become accepted candidates.

### 4. Our evaluation answers a smaller question

The article reports 196 LAB tasks (110 training, 86 validation), ten cycles, and
familiar/new-matter cohorts. It reports all-pass increasing from 2.9% to 15.7%
(+12.8 percentage points, about 5.4 times), and selective retrieval roughly halving
cost at comparable rubric quality. These are external author-reported results.
They should not be substituted with the separate Managed Agents ~6× claim.

Our [rollout script](../scripts/eval_rollouts.py) uses authored support fixtures,
a small synthetic task set, a local solver and deterministic action-effect checks.
It freezes memory for queries, which is useful infrastructure. It does not learn
memory through ten fresh rollout cycles. Existing arms vary admission as well as
retrieval and cannot isolate the cost/quality effect of full versus selected memory.

**Needed:** first build a local, reproducible multi-cycle experiment with disjoint
support/query tasks and a new-task-family split. Add no-memory, full-memory and
selected-memory arms that share the same frozen bank, solver and grading setup.
Measure all-pass, criterion pass rate, gains/harms, token use, latency, tool calls,
and total compute including reflection and gating. Legal LAB replication is a
separate later evaluation; a local proxy is not a reproduction of Luna/LAB results.

### 5. Pending traces are not an actionable review queue

Uncertain candidates are recorded in audit events and can be retried. Rejected
outcomes can be stored, but there is no dedicated review workflow that enumerates
items, records reviewer decisions and safely resubmits corrected candidates.
The article explicitly describes rejected lessons entering a review queue.

**Needed:** explicit queue states, reasons and provenance, plus inspect/revise/retry
operations. Preserve the difference between rejection, uncertainty and provider error.

## Recommended implementation order

1. **Graded episode and cycle contracts.** Add criterion-level verdicts, deliverable
   references, task/matter IDs, grade visibility rules and versioned cycle manifests.
   Verify that held-out grades never enter training memory.
2. **Batch Sleep consolidation and commit policy.** Review existing memory with multiple
   support episodes; propose traceable add/revise/merge/archive operations; check
   independent support and identifying details; retain conditions and exceptions.
3. **Memory-scale handling and review queue.** Remove the all-active comparison bottleneck
   without silently skipping conflicts. Exercise retries and concurrent revisions.
4. **Closed-loop local evaluation.** Run actual solver → verifier → Sleep → frozen Wake
   cycles with local models. Add full-memory versus selected-memory ablations and
   familiar/new-task-family cohorts. Tune thresholds only on development data.
5. **Host automation and legal evaluation.** Add transcript/deliverable export and optional
   scheduling after the learning loop works. Connect LAB only with the correct data,
   harness and rubric protocol, reporting model differences from the original study.

Useful release evidence would include: memory gaining a supported exception across
cycles; rejection of a lesson backed only by duplicated copies of one task; a
reproducible reduction in harmful reuse; and cost/quality comparisons on held-out tasks.
None of these outcomes is established merely by adding the corresponding code.

## Positioning today

An accurate description is:

> A local-evaluable Wake–Sleep memory prototype inspired by Harvey's Jev-gated
> learning loop, with evidence-backed admission and task-specific recall.

Avoid claims of complete Harvey reproduction, proven legal-task gains, or validated
50% savings for this repository. See [our results](EVALUATION.md) and
[generated-lesson limitations](REVERIFICATION.md).
