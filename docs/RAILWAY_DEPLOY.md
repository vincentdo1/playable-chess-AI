# Railway deploy: v4-full model

The 173 MB v4 checkpoint can't ride in git (`model/` is git-ignored *and*
GitHub rejects any single file > 100 MB). So the weights live in a **Hugging
Face model repo** and the backend downloads them on first boot
(`load_model._maybe_fetch_from_hf`). Local dev is unaffected: if the `.pt`
already exists on disk, nothing is downloaded and `huggingface_hub` is never
imported.

On Railway the app serves **low-sim MCTS with a movetime cap** (CPU-only box),
which keeps per-move latency and RAM bounded while staying well above raw
policy strength.

---

## 1. One-time: upload the checkpoint to Hugging Face

Needs a free HF account and a **write** token
(https://huggingface.co/settings/tokens). Run locally from the repo root:

```bash
.venv/Scripts/python.exe -m pip install huggingface_hub
huggingface-cli login                       # paste the write token
huggingface-cli repo create magnus-chess-v4 --type model
# upload  <repo_id>                      <local file>                              <path in repo>
huggingface-cli upload <hf-username>/magnus-chess-v4 model/grandmaster_resnet_v4_full.pt grandmaster_resnet_v4_full.pt
```

Leave the repo **public** (default) so no token is needed at serve time. Make
it private only if you also set `HF_TOKEN` on Railway.

To ship a new model later, upload a new file (or a new revision) and bump
`MAGNUS_HF_FILENAME` / `MAGNUS_HF_REVISION` on Railway.

## 2. Railway environment variables

Set these on the service (Railway dashboard -> Variables, or
`railway variables set KEY=VALUE`):

| Variable | Value | Why |
|---|---|---|
| `MODEL_PATH` | `model/grandmaster_resnet_v4_full.pt` | selects v4; also the local target path / default HF filename |
| `MAGNUS_HF_REPO` | `<hf-username>/magnus-chess-v4` | triggers the boot download |
| `MAGNUS_USE_MCTS` | `1` | search on |
| `MAGNUS_MCTS_SIMULATIONS` | `64` | low sims for CPU |
| `MAGNUS_MCTS_TIME_LIMIT` | `3` | hard 3 s/move cap (whichever hits first) |
| `MAGNUS_BLUNDER_GUARD` | `0` | the guard hurts v4 (and MCTS bypasses it anyway) |
| `MAGNUS_TEMPERATURE` | `0.0` | greedy at the root |

Optional: `MAGNUS_HF_FILENAME` (defaults to the basename of `MODEL_PATH`),
`MAGNUS_HF_REVISION` (defaults to latest), `HF_TOKEN` (private repos only),
`MAGNUS_MCTS_MAX_SIMULATIONS` (per-request ceiling, default 800).

## 3. Deploy

`Procfile` already runs `web: python -m backend.app` and `requirements.txt`
pins CPU torch + `huggingface_hub`. Push the branch (or `railway up`) and watch
the deploy logs for:

```
Model 'model/grandmaster_resnet_v4_full.pt' not found locally; fetching
  'grandmaster_resnet_v4_full.pt' from Hugging Face repo '<hf-username>/magnus-chess-v4'...
Downloaded checkpoint to /root/.cache/huggingface/...
Model loaded from ...  (encoding: perspective_v3)
  Saved at epoch 2  |  val_loss=1.3826
Magnus MCTS enabled: 64 sims/move
```

Then hit `GET /` — it should report
`"model": "grandmaster_resnet_v4_full.pt"`, `"use_mcts": true`,
`"mcts_simulations": 64`, `"mcts_time_limit": 3`, `"blunder_guard": false`.

## 4. Notes / tradeoffs

- **Cold-start cost:** Railway's filesystem is ephemeral, so the 173 MB pull
  happens on every cold start / redeploy (adds a few seconds to first boot).
  To persist it, attach a Railway volume and set `HF_HOME=/data/hf-cache`.
- **RAM:** CPU torch + the 15 M-param net + a small MCTS tree is roughly
  0.7-1.2 GB. Fine on Hobby (up to 8 GB); may be tight on the smallest trial.
- **Tuning:** if moves feel slow, lower `MAGNUS_MCTS_TIME_LIMIT` or
  `MAGNUS_MCTS_SIMULATIONS`; if you upgrade to a bigger Railway plan you can
  raise sims toward the 200 that scored 68.8% vs UCI_Elo 2500 locally.
- **Rollback:** unset `MODEL_PATH`/`MAGNUS_*` to fall back to the in-repo
  default (v3) — v2/v3 remain servable; only the env selects v4.
