# Chess AI - Vincent Do

A playable chess application with multiple AI opponents, including a perspective ResNet neural network (Magnus Carlsen NN) with optional MCTS search at inference. Runs as a browser app (Flask backend + static frontend) and as a Pygame desktop app.

**Live site:** https://vincentdo1.github.io/playable-chess-AI
**Backend (Railway):** drives the live site for Alphabeta and Magnus.

> GitHub Pages serves the `main` branch. Changes on other branches won't appear on the live site until merged to `main`.

<p align="center">
  <img src="media/network.gif" alt="CNN+LSTM forward pass" width="720">
  <br/>
  <em>Forward pass of the trained CNN + LSTM on a real position. <a href="media/network.mp4">MP4</a></em>
</p>

---

## Players

- **Random** — picks a legal move at random.
- **Alphabeta** — minimax + alpha-beta pruning + endgame-aware heuristics.
- **Stockfish** — UCI engine (desktop) or WebAssembly (browser).
- **Magnus Carlsen NN** — perspective ResNet with policy and value heads, trained on Magnus and GM games (v3 conv-head by default; the older v2 ResNet+LSTM checkpoints stay loadable). Optionally wrapped in MCTS at inference for stronger, more tactical play.

---

## Quick start (local play)

```powershell
# 1. Python 3.12 venv + dependencies
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # /cpu if no GPU
pip install -r requirements-local.txt

# 2. Put a trained model in model/ (gitignored; you provide it)
#    Default expected path: model/grandmaster_resnet_v3.pt
#    Either train it (see "Training") or copy in an existing checkpoint.
#    Old v2 checkpoints also work: point MODEL_PATH at them.

# 3. Verify the environment
$env:REQUIRE_CUDA = "1"     # "0" for CPU
python -m training.check_training_env

# 4. Start the backend (serves model/grandmaster_resnet_v3.pt by default)
$env:MAGNUS_USE_MCTS = "1"
$env:MAGNUS_MCTS_SIMULATIONS = "200"
python app.py

# 5. In a second terminal, serve the frontend
            
```

Open `http://localhost:8000`. If Magnus is greyed out, the backend failed to load the model — check the backend terminal for `Magnus model unavailable: ...` and the listed reason.

---

## Repository layout

```
playable-chess-AI/
  app.py                   Flask backend entry (used by Procfile)
  main.py                  Pygame desktop entry
  neural_network.py        Model architecture + supervised trainer
  load_model.py            Checkpoint loading + raw-policy inference
  chess_player.py          Non-NN players (alphabeta, random, Stockfish)
  heuristics.py            Piece-square tables for chess_player
  index.html               Frontend shell for GitHub Pages
  stockfish.js             Stockfish WASM for browser play
  Procfile                 Railway entry: `python -m backend.app`

  backend/                 Flask API
    app.py                 Routes, model loading, MCTS toggle
  frontend/src/            Browser ES modules
  pieces/                  Chess piece PNGs (used by frontend + Pygame)
  tests/                   pytest unit tests

  inference/               NN-augmented inference
    mcts_player.py         PUCT/MCTS
    search_player.py       Alpha-beta with NN evaluator
    blunder_guard.py       Shallow-search veto of material-losing policy moves
  training/                Training pipeline
    preprocess.py          PGN -> .npz chunks
    dedup_positions.py     Cap duplicate (mostly opening) positions pre-retrain
    resume_training.py     Safe resumed supervised training
    check_training_env.py  CUDA/torch preflight
    extract_lichess_gm_vs_lower.py
  experiments/             Not production
    self_play.py           AlphaZero-style data generation
    train_self_play.py     AlphaZero-style iteration loop
  evaluation/              Match-based and test-set evaluation
    eval_arena.py          Paired-opening Elo with 95% CI
    evaluate_model.py      Held-out test-set metrics
    cross_model_match.py   Cross-branch model matches (subprocess)
    move_server.py         JSON line move server (companion)
    play_match.py          Headless engine matches, PGN output
  scripts/
    train_pipeline.sh      preprocess -> train -> self-play
    compare_checkpoints.ps1   Full Elo comparison matrix

  model/                   PyTorch checkpoints   (gitignored)
  data/                    Training chunks       (gitignored)
  extractions/             Raw PGN archives      (gitignored)
  analysis_games/          play_match PGN output (gitignored)
  eval_logs/               compare_checkpoints   (gitignored)
```

Subpackages use absolute import paths (`from inference.mcts_player import ...`). Run scripts inside subpackages with `python -m`, e.g. `python -m evaluation.eval_arena --model_a ...`.

---

## Training

Three stages, optionally chained by `scripts/train_pipeline.sh`.

### Architectures

Two model generations coexist; serving dispatches on the checkpoint's stored
`board_encoding`, so old checkpoints keep working:

| | v2 (`perspective_v2`) | v3 (`perspective_v3`, current default) |
|---|---|---|
| Board input | 17 channels | 20 channels (+ halfmove clock, 2 repetition planes) |
| Move history | LSTM over last 10 moves | none (history measured ~1% and caused a FEN-only serving skew) |
| Policy head | FC 128 → 20,480 flat vocab | conv → 76 move-type planes × 64 from-squares (vocab 4,864) |
| Value head | FC off shared 128-dim trunk | conv directly off tower features |
| Regularization | none | AdamW weight decay 5e-4, label smoothing 0.05, head dropout 0.1 |

Select with `ARCH_VERSION=v2|v3` (training only). Each architecture's
training defaults write to its own checkpoint file and read its own
`data/*_chunks*` dirs, so neither generation ever overwrites the other.
Serving defaults to `model/grandmaster_resnet_v3.pt` — measured +137 Elo
(95% CI [+40, +261]) over the resumed v2 in greedy policy mode — and
dispatches on the checkpoint's stored encoding, so `MODEL_PATH` can point
at any v2 or v3 checkpoint.

### 1. Preprocess

```powershell
python -m training.preprocess              # v3 chunks -> data/*_chunks_v3
python -m training.preprocess --no_cp_loss # game-result value targets only
$env:PREPROCESS_ENCODING = "perspective_v2"  # legacy v2 chunks if needed
```

Reads `extractions/GM_games_2600.zip` and `extractions/magnus.zip`, writes `data/{train,val,test}_chunks_v3/`.

Optionally cap duplicated opening positions before training (82% of v2
opening samples were repeats, which starves middlegame learning):

```powershell
python -m training.dedup_positions --input_dir data/train_chunks_v3 `
  --output_dir data/train_chunks_v3_dedup --max_repeats 4
```

Adding Lichess strong-vs-weak as negatives:

```powershell
python -m training.extract_lichess_gm_vs_lower `
  --input "lichess_db_standard_rated_2026-04.pgn.zst" `
  --output "extractions\lichess_2500_vs_u2200.pgn" `
  --gm_min_elo 2500 --opponent_max_elo 2200 --max_games 100000

python -m training.preprocess `
  --single_pgn "extractions\lichess_2500_vs_u2200.pgn" `
  --output_dir "data\train_chunks_lichess" `
  --policy_color_mode tagged
```

### 2. Supervised training

```powershell
$env:REQUIRE_CUDA = "1"
python neural_network.py
```

v3 by default: reads `data/{train,val}_chunks_v3`, writes
`model/grandmaster_resnet_v3.pt`. Set `ARCH_VERSION=v2` for the legacy
architecture (reads `data/{train,val}_chunks`, writes the v2 checkpoint);
`TRAIN_DIR`/`VAL_DIR`/`MODEL_PATH` override any of the paths.

Up to 50 epochs, early stopping, AMP + cuDNN benchmark enabled. Set `INIT_MODEL_PATH` to warm-start from a previous checkpoint (current architecture only).

### 3. Resume / fine-tune

If Stage 2 stops early before convergence:

```powershell
python -m training.resume_training `
  --init   model\grandmaster_model_perspective_resnet_negatives_v2.pt `
  --output model\grandmaster_resnet_v2_resumed.pt `
  --epochs 40 --lr 3e-4 --early_stop_patience 10 --lr_patience 4
```

Only writes the output if val_loss strictly beats the source, so it cannot regress your current model.

### Pipeline shortcut

```powershell
bash scripts/train_pipeline.sh
```

---

## Evaluation

Three tools.

**`eval_arena.py`** — head-to-head Elo with 95% CI, paired opening suite:

```powershell
python -m evaluation.eval_arena `
  --model_a model\grandmaster_resnet_v3.pt `
  --model_b model\grandmaster_resnet_v2_resumed.pt `
  --method_a mcts --method_b mcts `
  --paired --games 128 --sims 200 --mcts_batch_size 16
```

(Greedy policy-mode baseline, 32 paired games, 2026-07-02: v3 beat
v2-resumed 17W-10D-5L, +137 Elo, 95% CI [+40, +261].)

Methods per side: `mcts`, `policy`, or `search`.

**`cross_model_match.py`** — subprocess-isolated matches for incompatible architectures (e.g. an old-branch checkpoint vs the current one). Needs `move_server.py` present in each worktree:

```powershell
git worktree add ..\playable-chess-AI-other other-branch
Copy-Item evaluation\move_server.py ..\playable-chess-AI-other\

python -m evaluation.cross_model_match `
  --player_a_dir ..\playable-chess-AI-other --player_a_model model\old.pt `
  --player_b_dir .                         --player_b_model model\new.pt `
  --games 20 --player_a_temperature 0.5 --player_b_temperature 0.5
```

**`compare_checkpoints.ps1`** — runs the full 4-way matrix (resumed vs base & vs self-play, in both mcts and policy):

```powershell
.\scripts\compare_checkpoints.ps1 `
  -Resumed  model\grandmaster_resnet_v2_resumed.pt `
  -Base     model\grandmaster_model_perspective_resnet_negatives_v2.pt `
  -SelfPlay model\selfplay_checkpoints\selfplay_iter0020.pt `
  -Sims 200 -Games 128
```

For held-out test-set metrics (defaults to the served checkpoint and its
encoding-matched `data/test_chunks[_v3]`; pass `--model`/`--test_dir` to
measure another one):

```powershell
python -m evaluation.evaluate_model --examples 10
```

**Phase diagnostics** — why does play degrade as the game goes on?

```powershell
# Policy/value quality by move-number bucket; also quantifies the
# FEN-only (zero move history) serving skew vs eval conditions.
python -m evaluation.diagnose_phase_degradation --max_positions 150000

# Stockfish cp-loss of the move the live backend would actually play,
# policy-only vs value-reranked, per phase.
python -m evaluation.diagnose_value_rerank --per_phase 150

# Same protocol, measuring the inference/blunder_guard.py effect.
python -m evaluation.diagnose_blunder_guard --per_phase 150 --guard_depth 2
```

---

## Self-play (experimental)

AlphaZero-style: the network plays itself with MCTS, training targets are MCTS visit distributions and game outcomes.

```powershell
python -m experiments.train_self_play `
  --init_checkpoint model\grandmaster_resnet_v3.pt `
  --iterations 20 `
  --mcts_simulations 400 --mcts_batch_size 16
```

The first attempt (LR 1e-3, 1000 steps/iter, 50 games/iter, self-play-only
buffer) collapsed from catastrophic forgetting — iter20 lost 20-0 to its
base. The defaults now implement the retry recipe:

- LR `1e-4`, `100` training steps/iter, `200` games/iter
- supervised chunks mixed into every batch (`--supervised_dirs`,
  default `data/train_chunks_v3`, 50/50 via `--supervised_fraction`)
- after each iteration a `--gate_games` policy-mode match runs against the
  pre-iteration weights; scoring under `--gate_min_score` (0.45) reverts the
  weights and stops the run. Failed iterations write no checkpoint.

Still don't deploy a self-play checkpoint without beating the base in `eval_arena`.

---

## Backend API

`backend/app.py` is the Flask app served on port 5000. Root `app.py` is the Procfile-compatible entry.

```powershell
# MODEL_PATH defaults to model/grandmaster_resnet_v3.pt; set it to serve
# a different checkpoint (v2 files still load).
$env:MAGNUS_USE_MCTS    = "1"
$env:MAGNUS_MCTS_SIMULATIONS = "200"
python app.py
```

**Endpoints:**

`GET /` — health. Returns `players.magnus: bool` and a `magnus` block with the loaded model name, defaults, and `mcts_available` / `use_mcts`.

`POST /api/move` — get the next move.

```json
{ "fen": "<FEN>", "player": "alphabeta", "depth": 3 }
{ "fen": "<FEN>", "player": "magnus", "temperature": 0.0, "value_weight": 2.0, "value_candidates": 0 }
{ "fen": "<FEN>", "player": "magnus", "use_mcts": true, "mcts_simulations": 200 }
```

Per-request fields override the server's env defaults. The response includes `"method": "policy"` or `"method": "mcts"`.

**Backend env vars:**

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `model/grandmaster_resnet_v3.pt` | Path to checkpoint (v2 checkpoints also load) |
| `MAGNUS_TEMPERATURE` | `0.0` | Policy sampling temperature |
| `MAGNUS_VALUE_WEIGHT` | `2.0` | Value-head reranking weight (policy mode) |
| `MAGNUS_VALUE_CANDIDATES` | `0` | Top-K policy candidates to value-check (0 = all) |
| `MAGNUS_BLUNDER_GUARD` | `1` | Shallow-search veto of material-losing policy moves (policy mode) |
| `MAGNUS_BLUNDER_GUARD_DEPTH` | `2` | Guard search depth (2 ≈ 20 ms/move, 3 ≈ 130 ms/move) |
| `MAGNUS_BLUNDER_GUARD_MARGIN` | `150` | Veto candidates this many cp worse than the best candidate |
| `MAGNUS_USE_MCTS` | `0` | Enable MCTS globally |
| `MAGNUS_MCTS_SIMULATIONS` | `200` | Sims per move |
| `MAGNUS_MCTS_BATCH` | `16` | Leaf-eval batch size |
| `MAGNUS_MCTS_C_PUCT` | `1.5` | PUCT exploration constant |
| `MAGNUS_MCTS_POLICY_TEMP` | `1.5` | Policy-prior softening |

---

## Desktop app (Pygame)

Loads the model directly (no backend); `MODEL_PATH` defaults to
`model/grandmaster_resnet_v3.pt`:

```powershell
python main.py --black_player magnus_carlsen
```

`--white_player` / `--black_player` options: `you`, `random`, `alphabeta`, `engine` (Stockfish), `magnus_carlsen`.

---

## Production deployment

The live frontend (GitHub Pages, served from `main`) calls a Flask backend on Railway. The frontend's production API URL is hard-coded in `frontend/src/config.js`.

To redeploy after model or backend changes:

1. **Push to `main`** — GitHub Pages picks up the frontend. Railway picks up the backend.
2. **Make sure the model file exists on Railway.** Because `model/` is gitignored, `git push` does **not** ship the checkpoint. Options:
   - **Railway Volume** (recommended): create a persistent volume mounted at `/app/model/`, upload your `.pt` file once via the Railway shell (`cat > model/grandmaster_resnet_v3.pt` with the file streamed in), and set `MODEL_PATH` to that path.
   - **Build-time download**: have your start command pull the model from an external URL (S3, Hugging Face, etc.) before launching the Flask app.
   - **Force-commit via git LFS**: track `.pt` files via Git LFS and remove `model/` from `.gitignore`. Simple but couples the model lifecycle to git history.
3. **Set Railway env vars**: `MODEL_PATH`, optionally `MAGNUS_USE_MCTS=1` / `MAGNUS_MCTS_SIMULATIONS=200` for stronger play.
4. **Verify deployment**: `curl https://<your-railway-url>/` and confirm `players.magnus: true` and `mcts_available: true` in the response. If `players.magnus: false`, the model didn't load — check Railway logs for the `Magnus model unavailable: <reason>` line.

If you didn't redeploy after a model architecture change, the Railway server may still hold a stale checkpoint (or none) and the frontend will hide the Magnus option even though local works.

---

## Tests

```powershell
python -m pytest tests/ -q
```
