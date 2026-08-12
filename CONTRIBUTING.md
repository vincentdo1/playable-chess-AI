# Contributing

Keep pull requests small enough to review as one coherent change. Explain the
problem and non-goals before the implementation, include exact validation
commands, and separate measured results from hypotheses or roadmap targets.

For ML changes, record immutable identities for the dataset snapshot, code,
checkpoint, evaluation engine, and opening suite. Include the relevant
training configuration and random seeds, uncertainty on strength metrics, and
p50/p95 latency for the configuration intended for production. Do not equate
a result against a limited-strength engine setting with a human rating.

Run the artifact-independent suite before opening a PR:

```bash
python -m pip install --requirement requirements-test.txt
python -m pytest -q
```

Tests that require ignored checkpoints, datasets, or a Stockfish binary skip
in a clean checkout. Run those integration audits locally with the documented
artifacts and attach their machine-readable output to the PR.

Changes that alter model inputs, checkpoint schemas, or public API behavior
need backward-compatibility tests. Deployment changes need a pinned rollout
artifact, readiness check, and a tested known-good rollback.

The repository does not currently declare a project license. Do not assume
permission to redistribute project code or model artifacts until the owner
selects one. Preserve third-party notices and dataset/model terms.
