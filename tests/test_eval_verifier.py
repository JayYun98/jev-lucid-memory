import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rollouts", Path(__file__).resolve().parents[1] / "scripts/eval_rollouts.py"
)
rollouts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollouts)


def test_verifier_checks_data_and_side_effects():
    q = {x["id"]: x for x in rollouts.queries()}
    assert rollouts.execute(
        q["pagination-all"], {"steps": [{"op": "fetch"}, {"op": "fetch", "cursor": "c1"}]}
    )
    assert not rollouts.execute(
        q["pagination-first"], {"steps": [{"op": "fetch"}, {"op": "fetch", "cursor": "c1"}]}
    )
    assert rollouts.execute(q["retry-read"], {"steps": [{"op": "read"}, {"op": "read"}]})
    assert not rollouts.execute(q["retry-read"], {"steps": [{"op": "read"}]})
    assert rollouts.execute(q["retry-write"], {"steps": [{"op": "status"}]})
    assert not rollouts.execute(q["retry-write"], {"steps": [{"op": "write"}, {"op": "status"}]})
    assert rollouts.execute(q["version-v1"], {"steps": [{"op": "list_records"}, {"op": "reverse"}]})
    assert not rollouts.execute(
        q["version-v2"], {"steps": [{"op": "list_records"}, {"op": "reverse"}]}
    )


def test_verifier_rejects_unknown_or_unbounded_actions():
    q = rollouts.queries()[0]
    for plan in ([], {}, {"steps": [{"op": "shell"}]}, {"steps": [{"op": "fetch"}] * 9}):
        assert not rollouts.execute(q, plan)
