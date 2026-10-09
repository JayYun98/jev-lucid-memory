"""Live local generated-lesson lifecycle check; semantic misses are reported, not hidden."""

import argparse
import json
import tempfile
from pathlib import Path

from jev_memory.core import Memory
from jev_memory.models import Episode, Task
from jev_memory.providers import LocalGate, LocalLLMGate, LocalWriter
from jev_memory.store import Store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
model = args.model
root = Path(__file__).resolve().parents[1]
episode = Episode.model_validate_json(
    (root / "examples/episodes.jsonl").read_text().splitlines()[0]
)
tasks = [
    Task.model_validate_json(x) for x in (root / "examples/tasks.jsonl").read_text().splitlines()
]
report = {}
with tempfile.TemporaryDirectory() as d:
    db = Path(d) / "memory.db"
    with Store(db) as s:
        m = Memory(s, LocalLLMGate(model=model), shadow=False)
        report["generated_sleep"] = m.sleep(episode, LocalWriter(model=model), trigger=False)
        assert s.lessons("demo"), report
    with Store(db) as s:
        m = Memory(s, LocalLLMGate(model=model), shadow=False)
        report["positive"] = m.wake(tasks[0])
        report["counterexample"] = m.wake(tasks[1])
        report["semantic_expectations"] = {
            "positive_selected": bool(report["positive"]["selected"]),
            "first_page_abstained": not bool(report["counterexample"]["selected"]),
        }
        report["other_namespace"] = m.wake(tasks[0].model_copy(update={"namespace": "other"}))
        assert not report["other_namespace"]["selected"]
        report["clef_positive"] = Memory(s, LocalGate(), shadow=False).wake(tasks[0])
        report["clef_counterexample"] = Memory(s, LocalGate(), shadow=False).wake(tasks[1])
        lesson = s.lessons("demo")[0]
        s.transition("demo", lesson["id"], lesson["revision"], "archived")
    with Store(db) as s:
        report["after_archive"] = Memory(s, LocalLLMGate(model=model), shadow=False).wake(tasks[0])
        assert not report["after_archive"]["selected"]
        report["audit"] = s.events("demo")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "audit"}))
