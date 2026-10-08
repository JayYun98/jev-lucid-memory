# Contributing

Keep provider calls separate from deterministic validation and storage. Unknown
or malformed decisions must never silently activate lessons. Avoid adding cloud
fallbacks or executing model-generated code.

Run `uv sync --locked`, `uv run pytest`, `uv run ruff check src tests scripts`,
`uv run ruff format --check src tests scripts`, and `uv run python -m build`.
Use synthetic traces in tests; do not commit private episodes or API keys.

For model-facing changes, run the local evaluation scripts and report the model
revision, questions/policy, latency, tokens and failures. Do not tune on held-out
query outputs, call selection accuracy task success, or count mocked providers as
live inference. Changes to public benchmark protocols must be labeled adaptations.
