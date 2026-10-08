# Security and trust

- Do not upload credentials, real private traces or memory databases in issues.
- API keys are read from environment variables and never deliberately logged or persisted.
- Cloud providers receive episode/task content only when explicitly selected. There is no fallback.
- SQLite contains plaintext observations and lesson text. Shadow mode also persists audit data.
  Use a private directory and encrypted disk; file permissions are not encryption.
- Hashes detect modified evidence, not falsified evidence. Only a trusted host should
  create verifier observations. A model's success claim is not a verifier.
- Learned lessons are untrusted advice, never system instructions or executable code.
- Namespace checks are logical separation, not authorization for a public multi-tenant service.
  Protect the database and host configuration; do not accept arbitrary namespace claims from clients.
- The Clef development server is loopback-only and unauthenticated. Keep it local.
  Its release directory contains Python code; pin and review the upstream revision.

Report suspected vulnerabilities privately to the repository owner through GitHub.
Do not include active secrets or identifiable user data.
