"""Exercise a real Hermes plugin manager in an isolated temporary profile."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from jev_memory.core import Memory
from jev_memory.hermes import register
from jev_memory.models import Candidate, Episode
from jev_memory.providers import LocalLLMGate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hermes-checkout", type=Path, required=True)
p.add_argument("--model", required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    os.environ["HERMES_HOME"] = d
    sys.path.insert(0, str(a.hermes_checkout.resolve()))
    import yaml
    from hermes_cli.plugins import PluginContext, PluginManager, parse_manifest_file

    from jev_memory.store import Store

    database = str(Path(d) / "memory.db")
    config = {
        "plugins": {
            "entries": {
                "jev-memory": {
                    "settings": {
                        "enabled": True,
                        "shadow": False,
                        "namespace": "test",
                        "database": database,
                        "provider": "local-llm",
                        "decision_model": a.model,
                        "tools": ["fetch"],
                        "environment": {"api": "v1"},
                    }
                }
            }
        }
    }
    Path(d, "config.yaml").write_text(yaml.safe_dump(config))
    scope = "test/" + hashlib.sha256(b"user-one").hexdigest()[:16]
    raw = json.loads((root / "examples/admission.jsonl").read_text().splitlines()[0])
    raw["episode"]["namespace"] = raw["candidate"]["namespace"] = scope
    with Store(database) as store:
        result = Memory(store, LocalLLMGate(model=a.model), shadow=False).admit_lesson(
            Episode.model_validate(raw["episode"]), Candidate.model_validate(raw["candidate"])
        )
        assert result["status"] == "active", result
    folder = root / "integrations/hermes"
    manifest = parse_manifest_file(folder / "plugin.yaml", folder, "user", "")
    assert manifest is not None
    manager = PluginManager(scope_key=d)
    register(PluginContext(manifest, manager))
    result = manager.invoke_hook(
        "pre_llm_call",
        user_message="Fetch all records with cursor pagination",
        session_id="test-session",
        sender_id="user-one",
        turn_id="one",
    )
    assert result and result[0]["context"].startswith("Retrieved experience"), result
    isolated = manager.invoke_hook(
        "pre_llm_call",
        user_message="Fetch all records with cursor pagination",
        session_id="test-session",
        sender_id="other-user",
        turn_id="two",
    )
    assert isolated == [], isolated
    print(
        json.dumps(
            {"real_plugin_manager": "pass", "local_inference": "pass", "sender_isolation": "pass"}
        )
    )
