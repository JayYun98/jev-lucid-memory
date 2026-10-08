"""Reproducible local-only smoke evaluation; not a general capability benchmark."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from jev_memory.core import RETRIEVAL, Memory
from jev_memory.models import Candidate, Episode, Task
from jev_memory.providers import LocalGate, LocalLLMGate, LocalWriter
from jev_memory.store import Store

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--provider", choices=["clef", "llm"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--writer-model", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
envelope = json.loads((root / "examples/admission.jsonl").read_text().splitlines()[0])
episode = Episode.model_validate(envelope["episode"])
candidate = Candidate.model_validate(envelope["candidate"])
gate = LocalGate(model=a.model) if a.provider == "clef" else LocalLLMGate(model=a.model)
# Fixed contrast set before any inference. Task IDs are held out from the source episode.
cases = [
    (
        "all",
        "Fetch all records with cursor pagination until the API returns next_cursor=null.",
        True,
    ),
    (
        "first",
        "Fetch only the first page of the cursor pagination API. Do not fetch more pages.",
        False,
    ),
    ("count", "Count all records from every page of the cursor pagination API.", True),
    (
        "explain",
        "Explain what cursor pagination means. Do not call the API or fetch records.",
        False,
    ),
    (
        "sample",
        "Fetch a sample of two records from the first page only, using cursor pagination.",
        False,
    ),
    (
        "export",
        "Export the complete dataset from the cursor pagination API, including every page.",
        True,
    ),
]
report = {
    "kind": "local smoke; six hand-authored contrasts, not a general benchmark",
    "provider": a.provider,
    "model": a.model,
    "threshold": 0.8,
    "external_api_charges_usd": 0,
    "hardware_cost_usd": None,
    "cases": [],
}
started = time.perf_counter()
with tempfile.TemporaryDirectory() as d, Store(Path(d) / "eval.db") as store:
    memory = Memory(store, gate, shadow=False)
    report["admission"] = memory.admit_lesson(episode, candidate)
    for id, text, expected in cases:
        task = Task(
            id="heldout-" + id,
            namespace=episode.namespace,
            text=text,
            environment=episode.environment,
            tools=episode.tools,
        )
        lexical = bool(store.search(task.namespace, text, 20))
        replay = gate.decide(
            {"task": task.model_dump(), "lesson": candidate.model_dump()}, RETRIEVAL
        )
        wake = memory.wake(task)
        report["cases"].append(
            {
                "id": id,
                "expected": expected,
                "no_memory": False,
                "retrieval_only": lexical,
                "shared_candidate_score": replay.scores["applicable"],
                "shared_candidate_replay": replay.scores["applicable"] >= 0.8,
                "shared_candidate_usage": replay.usage.model_dump(),
                "gated": bool(wake["selected"]),
                "latency_ms": wake["latency_ms"],
            }
        )
        print(id, bool(wake["selected"]), "expected", expected, flush=True)
    # Separate closed-loop smoke: writer output is not substituted into the shared-corpus test.
    with Store(Path(d) / "sleep.db") as sleeping:
        report["sleep"] = Memory(sleeping, gate, shadow=False).sleep(
            episode, LocalWriter(model=a.writer_model), trigger=False
        )
        report["sleep_events"] = sleeping.events(episode.namespace)
    report["events"] = store.events(episode.namespace)
report["latency_ms"] = (time.perf_counter() - started) * 1000
report["accuracy"] = {
    method: sum(c[method] == c["expected"] for c in report["cases"]) / len(cases)
    for method in ("no_memory", "retrieval_only", "gated", "shared_candidate_replay")
}
report["negative_transfer_proxy"] = sum(c["gated"] and not c["expected"] for c in report["cases"])
report["note"] = "Selection labels only: no downstream solver success or causal benefit measured."
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report["accuracy"]))
