# Railway rollout and rollback: v4 checkpoint

Status: serving path implemented; production deployment is not established by
tracked repository evidence.

The v4 checkpoint is gitignored and too large for a normal GitHub blob. The
backend can download it from a Hugging Face model repository during startup,
pin an immutable repository revision, verify a configured SHA-256 digest, load
it with PyTorch's restricted `weights_only` mode, and run a smoke forward pass.

This runbook deliberately separates “the code can deploy” from “this exact
model release was deployed and verified.” Do not mark the latter complete
without filling the release record in [`MODEL_CARD.md`](MODEL_CARD.md) and
capturing the rollout evidence below.

## 1. Prepare an immutable model release

Create or select a Hugging Face model repository and upload the checkpoint. A
typical manual flow is:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf upload <owner>/<model-repository> \
  model/grandmaster_resnet_v4_full.pt \
  grandmaster_resnet_v4_full.pt \
  --repo-type model
```

Do not paste a token into logs, source files, or Railway variable screenshots.
A private repository requires `HF_TOKEN` at runtime; a public repository does
not.

After upload:

1. Resolve the uploaded repository to an immutable commit SHA. Do not use
   `main` or “latest” for a production release.
2. Calculate the checkpoint's SHA-256 and exact byte count from the file that
   will be served.
3. Record the repository, commit, filename, byte count, digest, training code
   commit, and applicable model-weight license in the model card. This runbook
   does not choose a license.
4. Preserve the previous known-good artifact and its complete environment
   variable bundle for rollback.

Example digest commands:

```powershell
Get-FileHash model\grandmaster_resnet_v4_full.pt -Algorithm SHA256
(Get-Item model\grandmaster_resnet_v4_full.pt).Length
```

```bash
sha256sum model/grandmaster_resnet_v4_full.pt
wc -c model/grandmaster_resnet_v4_full.pt
```

## 2. Configure a staging service

Use a staging Railway service or environment before changing production. Set
actual observed values for placeholders:

| Variable | Staging value | Purpose |
|---|---|---|
| `MODEL_PATH` | `model/grandmaster_resnet_v4_full.pt` | Logical checkpoint path and default remote filename |
| `MAGNUS_HF_REPO` | `<owner>/<model-repository>` | Model artifact repository |
| `MAGNUS_HF_REVISION` | `<immutable-commit-sha>` | Required immutable download revision |
| `MAGNUS_MODEL_SHA256` | `<64-hex-checkpoint-digest>` | Fails startup if the downloaded/local file differs |
| `HF_TOKEN` | Railway secret, private repositories only | Read access to a private artifact |
| `MAGNUS_REQUIRED` | `1` | Treat missing model or required MCTS as a failed deployment |
| `MAGNUS_USE_MCTS` | `1` | Permit MCTS; clients may disable it but cannot enable it if this is `0` |
| `MAGNUS_MCTS_SIMULATIONS` | `64` | Server-owned simulation budget for the initial CPU rollout |
| `MAGNUS_MCTS_MAX_SIMULATIONS` | `64` | Hard ceiling; keep equal to the rollout budget unless deliberately tested |
| `MAGNUS_MCTS_TIME_LIMIT` | `3` | Soft wall-time cap in seconds; whichever budget is reached first stops search |
| `MAGNUS_MCTS_BATCH` | `16` | Leaf-evaluation batch size |
| `MAGNUS_BLUNDER_GUARD` | `0` | v4 MCTS does not use the legacy policy-mode guard |
| `MAGNUS_TEMPERATURE` | `0.0` | Greedy root selection |
| `INFERENCE_MAX_CONCURRENCY` | `1` | Bounds concurrent CPU-heavy inference; excess requests receive 429 |
| `MAX_REQUEST_BYTES` | `16384` | Caps JSON request bodies |
| `MAGNUS_ALLOWED_ORIGINS` | Exact production/staging frontend origins | CORS allowlist; do not use `*` in production |

The 64-simulation, three-second settings are a conservative operational
starting point, not a measured strength or latency claim. The historical 68.8%
point estimate used MCTS-200 on a local GPU; it does not establish the behavior
of this CPU configuration. It also predates the current v4 mask for unsupervised
clock/repetition input planes and therefore does not establish the strength of
the corrected inference path.

The browser now sends only the user's MCTS on/off choice. The server owns the
simulation budget, honors a lower client request, caps a higher request at the
effective server limit, and does not allow a client to turn on MCTS when the
server disabled it.

`MAGNUS_HF_REVISION` is required when `MAGNUS_HF_REPO` is set. The
`MAGNUS_ALLOW_FLOATING_HF_REVISION=1` escape hatch exists only for deliberate
local experimentation and should not be set in staging or production.

## 3. Deploy and verify staging

The `Procfile` runs one Gunicorn worker with two threads, a 180-second worker
timeout, and a 30-second graceful timeout so a cold model download/load has room
to finish. Those process timeouts are not the MCTS request budget; search is
still bounded separately by `MAGNUS_MCTS_TIME_LIMIT`. Application-level
inference concurrency remains bounded by `INFERENCE_MAX_CONCURRENCY`.

Deploy the branch and capture:

- Railway deployment ID and code commit;
- the complete non-secret variable values;
- model repository, immutable revision, filename, exact bytes, and SHA-256;
- cold-start duration and logs showing the resolved checkpoint, architecture,
  digest, and successful smoke forward;
- memory after startup and during representative moves;
- request latency and error/429 behavior under representative concurrency.

Use liveness and readiness for different purposes:

```bash
curl -fsS https://<staging-host>/livez
curl -fsS https://<staging-host>/readyz
```

- `GET /livez` returns 200 when the web process is alive.
- `GET /readyz` and the legacy `GET /` return 200 only when configured required
  capabilities are ready. A running but unready required service returns 503;
  an explicitly required model or MCTS import failure normally stops startup.

Confirm readiness reports `players.magnus: true`, `magnus.required: true`,
`magnus.ready: true`, `magnus.use_mcts: true`, the expected checkpoint basename,
architecture, checkpoint SHA-256, HF revision, simulation limit, and time limit.
Here, `magnus` is a legacy API field for the neutrally labeled neural-network
player. The response does not replace artifact verification: compare both its
SHA-256 and the startup-log SHA-256 with the recorded release digest.

Then submit a small set of valid FENs, including the initial position, a tactical
position, an endgame, and a FEN with a nonzero halfmove clock. (The FEN-only API
cannot carry repetition history.) This is the first release check of the real
checkpoint through the current masked v4 input path. For every response verify:

- HTTP 200 and a legal move;
- `method: "mcts"` when the request enables search;
- requested, server-budgeted, and actual simulation counts; elapsed time; and
  stop reason;
- the request stays within the configured wall-time objective;
- invalid payloads fail with 4xx rather than consuming unbounded work; and
- concurrent excess inference returns a bounded 429 with `Retry-After`.

Do not call staging verified from one legal opening move alone.

## 4. Promote to production

Promote the same code commit and the exact staging artifact/config bundle. Do
not re-upload the checkpoint under the same identity or switch to a floating
revision during promotion.

After promotion:

1. Check `/livez` and `/readyz` from outside Railway.
2. Repeat the legal-move smoke set against production.
3. Confirm the frontend can play a complete short interaction with MCTS both
   enabled and disabled.
4. Watch cold starts, 5xx/429 rates, p50/p95 move latency, memory, and restarts.
5. Record the deployment evidence and timestamp in
   [`PROJECT_STATUS.md`](PROJECT_STATUS.md) or a linked release record.

No production v4 deployment ID or verification result is currently recorded in
the repository.

## 5. Roll back

Preferred rollback: use Railway's rollback/redeploy capability to restore the
previous known-good deployment, including its code and environment variables.

Alternative rollback: restore a previously tested, complete artifact/config
bundle containing at least:

- `MODEL_PATH`;
- artifact repository, immutable revision, filename, and SHA-256;
- MCTS enabled flag, simulation/max-simulation budgets, batch size, and time
  cap;
- blunder-guard and policy settings;
- required/readiness, concurrency, request-size, and CORS settings.

After rollback, rerun readiness and the legal-move smoke set. Record why the
rollout was reverted and preserve the failed deployment's logs for diagnosis.

**Do not unset `MODEL_PATH` and `MAGNUS_*` and assume v3 exists in the image.**
The default v3 checkpoint is also gitignored, and Railway filesystems are
ephemeral. A usable rollback requires an actual prior artifact and configuration
or a previous successful Railway deployment.

## 6. Operational notes

- A cold instance downloads the model unless the Hugging Face cache is on a
  persistent Railway volume. `HF_HOME=/data/hf-cache` can point at such a cache,
  but the immutable revision and digest still need verification.
- A 15-million-parameter CPU model, PyTorch runtime, and MCTS tree can consume
  substantial memory. Earlier notes estimated roughly 0.7-1.2 GB, but measure
  the selected Railway plan rather than treating that estimate as an SLO.
- Lowering the search time or simulation budget may improve latency and reduce
  strength. Do not transfer the local MCTS-200 benchmark to another budget.
- Keep one Gunicorn worker unless memory and model-loading behavior have been
  measured with more. Thread count does not bypass the application inference
  semaphore.
- Treat `HF_TOKEN` as a secret and rotate it after suspected exposure.

## 7. Release evidence checklist

- [ ] Model card has repository, immutable revision, exact bytes, SHA-256, and
      model-weight license.
- [ ] Previous known-good deployment or artifact/config bundle is identified.
- [ ] Staging cold start loaded the pinned artifact and passed readiness.
- [ ] Canonical FEN smoke set returned legal moves.
- [ ] Latency, memory, concurrency, and error behavior were observed.
- [ ] Production used the exact staged code/artifact/config bundle.
- [ ] Production liveness, readiness, and frontend interaction passed.
- [ ] Deployment ID, timestamp, evidence links, and rollback target were
      recorded.
