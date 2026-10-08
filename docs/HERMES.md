# Hermes integration

The adapter is opt-in and only implements `pre_llm_call`. Install `jev-memory`
into the same Python environment as Hermes, then copy `integrations/hermes` to
`$HERMES_HOME/plugins/jev-memory`. Start with an isolated profile.

Example `config.yaml` fragment:

```yaml
auxiliary:
  background_review:
    enabled: false
plugins:
  entries:
    jev-memory:
      settings:
        enabled: true
        shadow: true
        namespace: my-project
        database: /absolute/private/path/memory.db
        provider: local
        decision_model: clef-flash
        endpoint: http://127.0.0.1:8766/v1/systemone
        tools: [fetch]
        environment: {api: v1}
```

Use `provider: local-llm` and the chat endpoint/model to use a generative local gate.
Use `typesafe` or `openrouter` only when intentionally allowing cloud inference.
`decision_model` avoids Hermes's reserved `model` configuration root.
Change `shadow` to `false` only after inspecting local audit results.

Memory is scoped to `namespace + '/' + sha256(sender_id)[:16]`; when the sender is
absent it uses `session_id`. Without either identity it skips. Support episodes
imported through the CLI must use that same scope. Stable sender IDs allow
cross-session reuse; session fallback intentionally does not. The operator must
ensure sender IDs are trusted and distinct across relevant transports.

Only text requests are supported. Tool availability and environment facts are
operator configuration, not auto-discovered. At most three candidates are checked,
with a five-second timeout per decision request. A failure skips memory and leaves
the host's main request intact. This is not a strict five-second total deadline.

There is no interception of Hermes's built-in reviewer writes. Disable built-in
background review only in your evaluation profile to compare memory systems fairly.
Sleep runs separately on explicit JSONL episodes; no unmerged upstream Sleep hook
is assumed. This adapter does not automatically export Hermes conversations.

## Reproduce the real-host check

```bash
uv run python scripts/check_hermes.py --hermes-checkout /path/to/hermes-agent \
  --model "$MEMORY_MODEL"
```

The script uses Hermes's real manifest parser, configuration reader, registration
and hook dispatcher in a temporary profile, plus real local model inference.
It verifies context injection and rejection for a different sender. It does not
run a complete interactive Hermes agent/solver session or alter your live profile.
Tested upstream revision is recorded in [IMPLEMENTATION.md](IMPLEMENTATION.md).
