# Chess AI - Vincent Do

A playable chess application with multiple AI opponents, including a perspective ResNet + LSTM neural network (Magnus Carlsen NN) with optional MCTS search at inference. Runs as a browser app (Flask backend + static frontend) and as a Pygame desktop app.

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
- **Magnus Carlsen NN** — perspective ResNet + LSTM with policy and value heads, trained on Magnus and GM games. Optionally wrapped in MCTS at inference for stronger, more tactical play.

---

## Quick start (local play)

```powershell
# 1. Python 3.12 venv + dependencies
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # /cpu if no GPU
pip install -r requirements-local.txt

# 2. Put a trained model in model/ (gitignored; you provide it)
#    Default expected path: model/grandmaster_model_perspective_resnet_negatives_v2.pt
#    Either train it (see "Training") or copy in an existing checkpoint.

# 3. Verify the environment
$env:REQUIRE_CUDA = "1"     # "0" for CPU
python -m training.check_training_env

# 4. Start the backend
$env:MODEL_PATH = "model\grandmaster_model_perspective_resnet_negatives_v2.pt"
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
  training/                Training pipeline
    preprocess.py          PGN -> .npz chunks
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

### 1. Preprocess

```powershell
python -m training.preprocess              # uses Stockfish if available
python -m training.preprocess --no_cp_loss # game-result value targets only
```

Reads `extractions/GM_games_2600.zip` and `extractions/magnus.zip`, writes `data/{train,val,test}_chunks/`. The board tensor has 17 perspective-relative channels.

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
$env:TRAIN_DIR    = "data/train_chunks"
$env:VAL_DIR      = "data/val_chunks"
$env:MODEL_PATH   = "model/grandmaster_model_perspective_resnet_negatives_v2.pt"
$env:REQUIRE_CUDA = "1"
python neural_network.py
```

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
  --model_a model\grandmaster_resnet_v2_resumed.pt `
  --model_b model\grandmaster_model_perspective_resnet_negatives_v2.pt `
  --method_a mcts --method_b mcts `
  --paired --games 128 --sims 200 --mcts_batch_size 16
```

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

For held-out test-set metrics:

```powershell
python -m evaluation.evaluate_model --model $env:MODEL_PATH --examples 10
```

---

## Self-play (experimental)

AlphaZero-style: the network plays itself with MCTS, training targets are MCTS visit distributions and game outcomes.

```powershell
python -m experiments.train_self_play `
  --init_checkpoint model\grandmaster_model_perspective_resnet_negatives_v2.pt `
  --iterations 20 --games_per_iteration 50 --training_steps 1000 `
  --mcts_simulations 400 --mcts_batch_size 16
```

In our runs this caused catastrophic forgetting (iter20 much weaker than the supervised base). Kept for future experiments with lower LR, fewer training steps per iteration, and supervised data mixed into the replay buffer. Don't deploy a self-play checkpoint without beating the base in `eval_arena`.

---

## Backend API

`backend/app.py` is the Flask app served on port 5000. Root `app.py` is the Procfile-compatible entry.

```powershell
$env:MODEL_PATH         = "model/grandmaster_model_perspective_resnet_negatives_v2.pt"
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
| `MODEL_PATH` | `model/grandmaster_model_perspective_resnet_negatives_v2.pt` | Path to checkpoint |
| `MAGNUS_TEMPERATURE` | `0.0` | Policy sampling temperature |
| `MAGNUS_VALUE_WEIGHT` | `2.0` | Value-head reranking weight (policy mode) |
| `MAGNUS_VALUE_CANDIDATES` | `0` | Top-K policy candidates to value-check (0 = all) |
| `MAGNUS_USE_MCTS` | `0` | Enable MCTS globally |
| `MAGNUS_MCTS_SIMULATIONS` | `200` | Sims per move |
| `MAGNUS_MCTS_BATCH` | `16` | Leaf-eval batch size |
| `MAGNUS_MCTS_C_PUCT` | `1.5` | PUCT exploration constant |
| `MAGNUS_MCTS_POLICY_TEMP` | `1.5` | Policy-prior softening |

---

## Desktop app (Pygame)

Loads the model directly (no backend). Set `MODEL_PATH` first if Magnus is playing:

```powershell
$env:MODEL_PATH = "model\grandmaster_model_perspective_resnet_negatives_v2.pt"
python main.py --black_player magnus_carlsen
```

`--white_player` / `--black_player` options: `you`, `random`, `alphabeta`, `engine` (Stockfish), `magnus_carlsen`.

---

## Production deployment

The live frontend (GitHub Pages, served from `main`) calls a Flask backend on Railway. The frontend's production API URL is hard-coded in `frontend/src/config.js`.

To redeploy after model or backend changes:

1. **Push to `main`** — GitHub Pages picks up the frontend. Railway picks up the backend.
2. **Make sure the model file exists on Railway.** Because `model/` is gitignored, `git push` does **not** ship the checkpoint. Options:
   - **Railway Volume** (recommended): create a persistent volume mounted at `/app/model/`, upload your `.pt` file once via the Railway shell (`cat > model/grandmaster_model_perspective_resnet_negatives_v2.pt` with the file streamed in), and set `MODEL_PATH` to that path.
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
