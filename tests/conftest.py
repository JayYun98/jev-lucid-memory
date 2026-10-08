import pytest

from jev_memory.models import Candidate, Decision, Episode, Observation, Task
from jev_memory.store import Store


class ScriptedGate:
    """A test double, never a measured model or a production fallback."""

    model = "scripted-test-only"

    def __init__(self, **scores):
        self.scores = scores
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return Decision(
            model=self.model,
            scores={
                k: self.scores.get(
                    k, 0.01 if k.startswith(("duplicate_", "contradiction_")) else 0.99
                )
                for k in questions
            },
        )


@pytest.fixture
def gate():
    return ScriptedGate()


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "memory.db") as s:
        yield s


@pytest.fixture
def episode():
    return Episode(
        id="episode-1",
        task_id="task-1",
        task_group="pagination",
        namespace="demo",
        task="Fetch all records from cursor pagination API",
        host_revision="test",
        solver_model="test",
        tools=["fetch"],
        environment={"api": "v1"},
        solver_claimed_success=True,
        verifier_result="pass",
        observations=[
            Observation.create(
                "tool-1",
                "tool",
                "Page one returned [1,2], next_cursor=2. Page two returned [3,4], next_cursor=null.",
            ),
            Observation.create("check-1", "verifier", "All four records collected; pass."),
        ],
    )


@pytest.fixture
def candidate(episode):
    return Candidate(
        namespace="demo",
        trigger="Fetch all records with cursor pagination",
        lesson="Follow next_cursor until it is null when all records are requested.",
        preconditions=["The API uses next_cursor; all records are requested."],
        exceptions=["Only the first page is requested."],
        required_tools=["fetch"],
        environment={"api": "v1"},
        evidence={o.id: o.sha256 for o in episode.observations},
    )


@pytest.fixture
def task():
    return Task(
        id="new-task",
        namespace="demo",
        text="Fetch all records with cursor pagination",
        tools=["fetch"],
        environment={"api": "v1"},
    )
