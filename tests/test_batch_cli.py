import json

from jev_memory import cli
from jev_memory.consolidation import EvidenceRef, GradedEpisode, MemoryOperation
from jev_memory.models import Observation, canonical
from jev_memory.store import Store


class Reviewer:
    model = "test-reviewer"

    def __init__(self, operations):
        self.operations = operations
        self.calls = 0

    def propose(self, episodes, snapshot):
        self.calls += 1
        return self.operations


def batch(episode, candidate):
    observation = Observation.create("tool-2", "tool", "Another matter verified cursor traversal.")
    second = episode.model_copy(
        update={
            "id": "episode-2",
            "task_id": "task-2",
            "observations": [observation],
            "verifier_result": "unknown",
        }
    )
    graded = [
        GradedEpisode(episode=episode, matter_id="matter-1"),
        GradedEpisode(episode=second, matter_id="matter-2"),
    ]
    refs = [
        EvidenceRef(
            episode_id=episode.id,
            observation_id=episode.observations[0].id,
            sha256=episode.observations[0].sha256,
        ),
        EvidenceRef(episode_id=second.id, observation_id=observation.id, sha256=observation.sha256),
    ]
    candidate = candidate.model_copy(
        update={"evidence": {ref.observation_id: ref.sha256 for ref in refs}}
    )
    return graded, MemoryOperation(action="add", candidate=candidate, evidence=refs)


def test_batch_cli_validates_whole_input_before_model_calls(
    tmp_path, monkeypatch, episode, candidate, capsys
):
    graded, operation = batch(episode, candidate)
    source = tmp_path / "batch.jsonl"
    source.write_text(canonical(graded[0]) + "\n{}\n")
    reviewer = Reviewer([operation])
    monkeypatch.setattr(cli, "BatchWriter", lambda writer: reviewer)
    monkeypatch.setattr(
        cli, "LocalGate", lambda **kw: (_ for _ in ()).throw(AssertionError("gate created"))
    )
    db = str(tmp_path / "memory.db")
    assert (
        cli.main(["--db", db, "--apply", "batch-sleep", str(source), "--cycle-id", "cycle-1"]) == 2
    )
    assert reviewer.calls == 0
    assert "Traceback" not in capsys.readouterr().err
    with Store(db) as store:
        assert store.lessons("demo") == []
        assert store.events("demo") == []
    source.write_text(canonical(graded[0]) + "\n" + canonical(graded[0]))
    assert (
        cli.main(["--db", db, "--apply", "batch-sleep", str(source), "--cycle-id", "cycle-1"]) == 2
    )
    assert reviewer.calls == 0


def test_batch_cli_shadow_apply_queue_and_retry(
    tmp_path, monkeypatch, gate, episode, candidate, capsys
):
    graded, operation = batch(episode, candidate)
    source = tmp_path / "batch.jsonl"
    source.write_text("\n".join(canonical(g) for g in graded))
    db = str(tmp_path / "memory.db")
    reviewer = Reviewer([operation])
    monkeypatch.setattr(cli, "BatchWriter", lambda writer: reviewer)
    monkeypatch.setattr(cli, "LocalGate", lambda **kw: gate)
    gate.scores["client_specific"] = 0.01
    prefix = ["--db", db]

    assert cli.main(prefix + ["batch-sleep", str(source), "--cycle-id", "shadow"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "shadow"
    with Store(db) as store:
        assert store.lessons("demo") == []

    gate.scores["supported"] = 0.5
    assert cli.main(prefix + ["--apply", "batch-sleep", str(source), "--cycle-id", "live"]) == 0
    pending = json.loads(capsys.readouterr().out)
    assert pending["status"] == "pending"
    queue_id = pending["queue_id"]

    monkeypatch.setattr(
        cli, "LocalGate", lambda **kw: (_ for _ in ()).throw(AssertionError("network provider"))
    )
    assert cli.main(prefix + ["queue", "--namespace", "demo"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"] == queue_id
    assert cli.main(prefix + ["--apply", "retry", str(queue_id), "--namespace", "other"]) == 2
    assert "queue entry unavailable" not in capsys.readouterr().err

    monkeypatch.setattr(cli, "LocalGate", lambda **kw: gate)
    gate.scores["supported"] = 0.99
    assert cli.main(prefix + ["--apply", "retry", str(queue_id), "--namespace", "demo"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "active"
    assert cli.main(prefix + ["--apply", "retry", str(queue_id), "--namespace", "demo"]) == 2
    capsys.readouterr()
    with Store(db) as store:
        assert len(store.lessons("demo")) == 1


def test_batch_cli_provider_failure_exit_one(tmp_path, monkeypatch, episode, candidate, capsys):
    graded, _ = batch(episode, candidate)
    source = tmp_path / "batch.jsonl"
    source.write_text("\n".join(canonical(g) for g in graded))

    class Broken:
        model = "broken"

        def propose(self, episodes, snapshot):
            raise ConnectionError("sensitive diagnostic")

    monkeypatch.setattr(cli, "BatchWriter", lambda writer: Broken())
    assert (
        cli.main(
            [
                "--db",
                str(tmp_path / "db"),
                "--apply",
                "batch-sleep",
                str(source),
                "--cycle-id",
                "broken",
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "reviewer_error"
    assert "sensitive" not in canonical(result)
