import json

import httpx
import pytest

from jev_memory.providers import LocalGate, LocalLLMGate, LocalWriter, OpenRouterGate, ProviderError


def transport_for(body, status=200):
    return httpx.MockTransport(lambda request: httpx.Response(status, json=body))


def test_decision_wire_contract():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/alpha/decisions"
        assert body["questions"]["supported"]["type"] == "noul"
        assert "Do observations support this?" in body["questions"]["supported"]["instructions"]
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13",
                "answers": {"supported": {"type": "noul", "noul": 0.92}},
                "usage": {"input_tokens": 100, "cost": 0.0000042},
            },
        )

    result = OpenRouterGate(api_key="test", transport=httpx.MockTransport(handler)).decide(
        {"text": "observations"}, {"supported": "Do observations support this?"}
    )
    assert result.scores == {"supported": 0.92}
    assert result.usage.cost_usd == 0.0000042


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"extra": {"type": "noul", "noul": 1}},
        {"a": {"type": "noul", "noul": "0.9"}},
        {"a": {"type": "noul", "noul": True}},
        {"a": {"type": "noul", "noul": 2}},
        {"a": {"type": "choice", "noul": 0.9}},
    ],
)
def test_malformed_answers_fail_closed(answers):
    gate = OpenRouterGate(api_key="test", transport=transport_for({"answers": answers}))
    with pytest.raises(ProviderError):
        gate.decide({}, {"a": "Question?"})


def test_provider_errors_do_not_leak_or_retry():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, text="API_KEY_SECRET")

    with pytest.raises(ProviderError) as exc:
        OpenRouterGate(api_key="KEY", transport=httpx.MockTransport(handle)).decide({}, {"a": "Q?"})
    assert "SECRET" not in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://10.0.0.1:8000",
        "http://localhost.evil.com",
        "http://user:pass@localhost:8000",
        "http://localhost:8000?q=secret",
    ],
)
def test_local_provider_cannot_leak_to_remote(url):
    with pytest.raises(ValueError):
        LocalGate(endpoint=url)


def test_local_gate_never_uses_cloud_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "PRIVATE")

    def handle(request):
        assert request.url.host == "127.0.0.1"
        assert "PRIVATE" not in str(request.headers)
        return httpx.Response(200, json={"answers": {"a": {"type": "noul", "noul": 0.99}}})

    assert (
        LocalGate(transport=httpx.MockTransport(handle)).decide({}, {"a": "Q?"}).scores["a"] == 0.99
    )


def test_writer_schema_and_empty_result(episode):
    body = {
        "choices": [{"message": {"content": '{"candidates":[]}'}}],
        "usage": {"prompt_tokens": 50},
    }
    candidates, usage = LocalWriter(transport=transport_for(body)).propose(episode)
    assert candidates == [] and usage.input_tokens == 50


def test_writer_rejects_extra_candidates(episode, candidate):
    body = {
        "choices": [
            {"message": {"content": json.dumps({"candidates": [candidate.model_dump()] * 4})}}
        ]
    }
    with pytest.raises(ProviderError):
        LocalWriter(transport=transport_for(body)).propose(episode)


def test_local_llm_gate_contract():
    body = {"choices": [{"message": {"content": '{"a":0.87}'}}]}
    assert LocalLLMGate(transport=transport_for(body)).decide({}, {"a": "Q?"}).scores["a"] == 0.87


def test_official_typesafe_sdk_contract(monkeypatch):
    import httpx2
    import typesafe_sdk

    from jev_memory.providers import TypeSafeGate

    real_client = typesafe_sdk.TypeSafeClient

    def handle(request):
        assert request.url.path == "/v1/systemone"
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"a": {"type": "noul", "noul": 0.91}},
                "usage": {"input_tokens": 20, "output_tokens": 0},
            },
        )

    monkeypatch.setattr(
        typesafe_sdk,
        "TypeSafeClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx2.MockTransport(handle)),
    )
    assert TypeSafeGate(api_key="test").decide({}, {"a": "Q?"}).scores == {"a": 0.91}
