import json

from jev_memory.cli import main
from jev_memory.hermes import register
from jev_memory.models import canonical


class Context:
    def __init__(self, settings):
        self.settings = settings
        self.hooks = {}

    def get_config(self, k, default=None):
        return self.settings.get(k, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_plugin_disabled_and_isolated(tmp_path, monkeypatch):
    ctx = Context({})
    register(ctx)
    assert ctx.hooks["pre_llm_call"](user_message="hello") is None
    ctx = Context({"enabled": True, "namespace": "team", "database": str(tmp_path / "db")})
    register(ctx)
    assert ctx.hooks["pre_llm_call"](user_message="hello") is None
    assert ctx.hooks["pre_llm_call"](user_message=[{"image": "x"}], session_id="s") is None
    from jev_memory import hermes

    seen = []

    def wake(self, task):
        seen.append(task)
        return {"context": "scoped memory"}

    monkeypatch.setattr(hermes.Memory, "wake", wake)
    result = ctx.hooks["pre_llm_call"](user_message="hello", session_id="s", sender_id="u1")
    assert result == {"context": "scoped memory"}
    ctx.hooks["pre_llm_call"](user_message="hello", session_id="s", sender_id="u2")
    assert seen[0].namespace != seen[1].namespace


def test_plugin_preserves_host_on_failure(tmp_path, monkeypatch):
    from jev_memory import hermes

    def broken(*a, **kw):
        raise RuntimeError("sensitive")

    monkeypatch.setattr(hermes, "LocalGate", broken)
    ctx = Context({"enabled": True, "namespace": "team", "database": str(tmp_path / "db")})
    register(ctx)
    assert ctx.hooks["pre_llm_call"](user_message="hello", session_id="s") is None


def test_cli_list_and_validate_all_before_writes(tmp_path, capsys, episode, candidate):
    db = str(tmp_path / "db")
    source = tmp_path / "in.jsonl"
    source.write_text(
        canonical({"episode": episode.model_dump(), "candidate": candidate.model_dump()}) + "\n{}\n"
    )
    assert main(["--db", db, "--apply", "admit", str(source)]) == 2
    assert main(["--db", db, "list", "--namespace", "demo"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_shadow_and_apply(tmp_path, monkeypatch, gate, episode, candidate, capsys):
    from jev_memory import cli

    monkeypatch.setattr(cli, "LocalGate", lambda **kw: gate)
    source = tmp_path / "in.jsonl"
    db = str(tmp_path / "db")
    source.write_text(
        canonical({"episode": episode.model_dump(), "candidate": candidate.model_dump()})
    )
    assert main(["--db", db, "admit", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["shadow"] is True
    assert main(["--db", db, "--apply", "admit", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "active"
    assert (
        main(
            [
                "--db",
                db,
                "--apply",
                "archive",
                "--namespace",
                "demo",
                result["id"],
                "--revision",
                "1",
            ]
        )
        == 0
    )


def test_sleep_reports_trigger_transport_failure(tmp_path, monkeypatch, episode, capsys):
    from jev_memory import cli

    class Broken:
        model = "offline"

        def decide(self, *args):
            raise ConnectionError("private diagnostic")

    monkeypatch.setattr(cli, "LocalGate", lambda **kw: Broken())
    source = tmp_path / "episode.jsonl"
    source.write_text(canonical(episode))
    assert main(["--db", str(tmp_path / "db"), "sleep", str(source)]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "provider_error"


def test_sleep_reports_nested_admission_failure(tmp_path, monkeypatch, episode, candidate, capsys):
    from jev_memory import cli
    from jev_memory.models import Usage

    class Broken:
        model = "offline"

        def decide(self, *args):
            raise ConnectionError("private diagnostic")

    class Writer:
        model = "test"

        def propose(self, ep):
            return [candidate], Usage()

    monkeypatch.setattr(cli, "LocalGate", lambda **kw: Broken())
    monkeypatch.setattr(cli, "LocalWriter", lambda **kw: Writer())
    source = tmp_path / "episode.jsonl"
    source.write_text(canonical(episode))
    assert main(["--db", str(tmp_path / "db"), "sleep", str(source), "--no-trigger"]) == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["reason"] == "provider_error"
