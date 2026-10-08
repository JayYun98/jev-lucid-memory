# Local inference

The default decision endpoint is `http://127.0.0.1:8766/v1/systemone`.
The default writer endpoint is `http://127.0.0.1:8123/v1/chat/completions`.
Local providers reject remote URLs and use no cloud credentials. No automatic fallback.

## Clef reference server

Allow at least 25 GB of free disk space for the BF16 release and dependencies,
and enough memory for both models plus their activations. The 64 GB Apple Silicon
validation machine runs the decision model and a separately quantized local writer.

```bash
uv sync --extra serve
uv run --extra serve python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Cloudflare/clef-flash',
    revision='fde727a287004204b7518dcc983fe64379776712'))
PY
# Substitute the snapshot path printed above.
uv run --extra serve python scripts/serve_clef.py --model-path /path/to/snapshot --device mps
```

Use `--device cuda` for compatible NVIDIA hardware or `--device cpu` for CPU.
Those device paths are configurable; only explicitly reported devices have been tested.
The pinned snapshot includes `joint_schema_model.py`, which is loaded as executable
Python. Review it before loading a different release. Never use an untrusted directory.

The server is text-only, binds loopback, serializes inference, and rejects requests
that exceed its token limit rather than silently truncating evidence. `/health`
reports the release directory and configured device. `--max-tokens` defaults to 4096.
Do not expose this development server to the public internet.

The official model card tested CUDA/H200. Our local MPS behavior is recorded in
[EVALUATION.md](EVALUATION.md), not assumed to match CUDA numerically.
The optional [community vllm-jev adapter](https://github.com/mode-io/vllm-jev) is not
bundled or validated here. Reference/adapter probability parity must be measured
before substituting it into a benchmark.

## Writer or generative gate

An OpenAI-compatible local server must support non-streaming chat completions.
The client requests JSON, temperature zero and a bounded output. Invalid JSON,
unknown evidence IDs and provider errors do not activate new lessons.
`--endpoint` overrides the decision endpoint; `sleep --writer-endpoint` overrides
the writer separately. Use explicit model IDs supported by that server.

Local API charges are zero. Electricity, model download, hardware time and
co-resident model contention are not free and are reported separately where measured.
