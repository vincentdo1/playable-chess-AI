# Railway deployment

The backend can download a gitignored checkpoint from Hugging Face at startup.
Use a pinned model revision and checksum so each deployment loads a known file.

## Prepare the model

Upload the checkpoint to a Hugging Face model repository:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf upload <owner>/<model-repository> \
  model/grandmaster_resnet_v4_full.pt \
  grandmaster_resnet_v4_full.pt \
  --repo-type model
```

Record the repository commit and calculate the file's SHA-256:

```powershell
Get-FileHash model\grandmaster_resnet_v4_full.pt -Algorithm SHA256
```

Do not store `HF_TOKEN` in the repository or deployment logs.

## Configure Railway

The `Procfile` starts `backend.app:app`. Set these variables on the Railway
service:

| Variable | Example |
| --- | --- |
| `MODEL_PATH` | `model/grandmaster_resnet_v4_full.pt` |
| `MAGNUS_HF_REPO` | `<owner>/<model-repository>` |
| `MAGNUS_HF_REVISION` | Immutable Hugging Face commit SHA |
| `MAGNUS_MODEL_SHA256` | Checkpoint SHA-256 |
| `HF_TOKEN` | Railway secret; private repositories only |
| `MAGNUS_REQUIRED` | `1` |
| `MAGNUS_USE_MCTS` | `1` |
| `MAGNUS_MCTS_SIMULATIONS` | `64` |
| `MAGNUS_MCTS_MAX_SIMULATIONS` | `64` |
| `MAGNUS_MCTS_TIME_LIMIT` | `3` |
| `INFERENCE_MAX_CONCURRENCY` | `1` |
| `MAGNUS_ALLOWED_ORIGINS` | Frontend origin |

The search values above are conservative CPU defaults, not the configuration
used by the historical MCTS-200 result. Measure the actual latency and memory
on the selected Railway plan.

`MAGNUS_HF_REVISION` is required when `MAGNUS_HF_REPO` is set.
`MAGNUS_ALLOW_FLOATING_HF_REVISION=1` is intended only for local experiments.

## Verify a deployment

Check both endpoints after a cold start:

```bash
curl -fsS https://<host>/livez
curl -fsS https://<host>/readyz
```

`/livez` reports that the process is running. `/readyz` reports whether the
configured model and search code loaded successfully. Confirm that readiness
shows the expected checkpoint, architecture, SHA-256, model revision, and MCTS
limits.

Before switching the frontend, test several valid FENs against `POST /api/move`
and confirm that every response contains a legal move. Also check startup time,
memory, request latency, 5xx responses, and concurrency limits.

## Roll back

Keep the previous working Railway deployment or its complete code, model, and
environment-variable configuration. If the release fails, redeploy that version
and repeat the readiness and legal-move checks.

Do not fall back by simply unsetting the model variables. Checkpoints are
gitignored and Railway filesystems are ephemeral, so a clean instance does not
contain the default v3 model.
