import json

import httpx
import pytest

from jev_memory.batch_writer import BatchWriter
from jev_memory.consolidation import GradedEpisode
from jev_memory.models import Observation
from jev_memory.providers import LocalWriter, ProviderError


def test_batch_writer_binds_host_hashes_and_rejects_invented_refs(episode):
    second = episode.model_copy(
        update={
            "id": "episode-2",
            "task_id": "task-2",
            "verifier_result": "unknown",
            "observations": [
                Observation.create("tool-1", "tool", "Another matter returned all pages.")
            ],
        }
    )
    episodes = [
        GradedEpisode(episode=episode, matter_id="a"),
        GradedEpisode(episode=second, matter_id="b"),
    ]
    proposal = {
        "operations": [
            {
                "action": "add",
                "candidate": {
                    "trigger": "Export all records",
                    "lesson": "Follow each page cursor.",
                    "preconditions": ["All records requested."],
                    "exceptions": ["First page only."],
                },
                "evidence": [
                    {"episode_id": e.episode.id, "observation_id": "tool-1"} for e in episodes
                ],
            }
        ]
    }

    def respond(request):
        body = json.loads(request.content)
        state = json.loads(body["messages"][1]["content"])
        assert len(state["support_episodes"]) == 2
        assert state["memory_snapshot"] == []
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(proposal)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
        )

    writer = BatchWriter(LocalWriter(transport=httpx.MockTransport(respond)))
    operation = writer.propose(episodes, [])[0]
    assert operation.candidate.namespace == "demo"
    assert operation.candidate.evidence == {
        "episode-1/tool-1": episode.observations[0].sha256,
        "episode-2/tool-1": second.observations[0].sha256,
    }
    assert writer.last_usage.input_tokens == 50
    proposal["operations"][0]["evidence"][0]["observation_id"] = "invented"
    with pytest.raises(ProviderError):
        writer.propose(episodes, [])
