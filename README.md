# Chess AI - Vincent Do

A playable chess application featuring multiple AI players including a neural network trained on Magnus Carlsen's games. Available as a desktop app and a web interface.

---

## Project overview

This project combines several approaches to chess AI:

- **Random** - picks a legal move at random
- **Alphabeta** - minimax search with alpha-beta pruning, endgame-aware heuristics, and move ordering
- **Stockfish** - UCI engine integration (desktop only)
- **Magnus Carlsen NN** - a perspective ResNet + LSTM neural network trained on played human moves from GM/Magnus PGNs, with a legal move-policy head, a position value head, optional Stockfish/search quality metadata, and temperature sampling for move variety

---

## Web interface

The easiest way to play. No setup required.

**Live site:** `https://vincentdo1.github.io/playable-chess-AI`

Features available in the browser:
- Human, Stockfish (WebAssembly), Random, Alphabeta AI, and Magnus Carlsen NN
- Adjustable Stockfish skill level (0-20)
- Adjustable alphabeta search depth
- Move history, status display, flip board, undo move

The web interface calls a Flask backend hosted on Railway for the Alphabeta and Magnus Carlsen players. Stockfish and Random run entirely in the browser with no server needed.

The frontend is split into a small HTML shell plus ES modules under `frontend/src/`:

- `api/` handles Flask calls.
- `components/` owns DOM-facing UI controllers.
- `game/` owns chess game orchestration.
- `services/` wraps browser engines such as Stockfish.
- `styles/` contains app CSS.

For local frontend development, serve the repo root instead of opening the file directly:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000`.

---

## Desktop app

### Requirements

- Python 3.12
- An Nvidia GPU (recommended for Magnus Carlsen NN inference)
- Stockfish 17 - download from `https://stockfishchess.org/download/`

### Installation

**Step 1 - Create a virtual environment with Python 3.12:**
```
py -3.12 -m venv chess_env
chess_env\Scripts\activate
```

**Step 2 - Install PyTorch with CUDA support:**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Step 3 - Install remaining dependencies:**
```
pip install -r requirements-local.txt
```

**Step 4 - Add Stockfish**

Place `stockfish.exe` in the project root, or set the environment variable to point to it anywhere on your machine:
```
set STOCKFISH_PATH=C:\path\to\stockfish.exe
```

**Step 5 - Add the trained model**

Place `grandmaster_model_perspective_resnet_negatives_v2.pt` in the `model/` folder:
```
model/grandmaster_model_perspective_resnet_negatives_v2.pt
```

The model is not included in the repository due to its size. Contact the project owner or retrain using the instructions below.

### Running the desktop app

Activate the virtual environment first:
```
chess_env\Scripts\activate
```

**Play as human vs human (default):**
```
python main.py
```

**Play as human vs a specific AI:**
```
python main.py --black_player random
python main.py --black_player alphabeta
python main.py --black_player engine
python main.py --black_player magnus_carlsen
```

The Magnus neural network uses policy+value move selection by default. Tune it
with:
```
python main.py --black_player magnus_carlsen --magnus_temperature 0.8 --magnus_value_weight 2.0 --magnus_value_candidates 0
```
`--magnus_value_weight 0` disables value reranking. `--magnus_value_candidates 0`
checks every legal move with the value head; a positive number checks only the
top policy candidates.

**Watch two AIs play each other:**
```
python main.py --white_player alphabeta --black_player random
python main.py --white_player engine --black_player alphabeta
python main.py --white_player magnus_carlsen --black_player engine
python main.py --white_player random --black_player magnus_carlsen
```

**Available player options:**

| Option | Description |
|---|---|
| `you` | Human player (default) |
| `random` | Picks a random legal move |
| `alphabeta` | Minimax with alpha-beta pruning and endgame heuristics |
| `engine` | Stockfish UCI engine |
| `magnus_carlsen` | Neural network trained on Magnus Carlsen's games |

---

## Training the neural network

### Step 1 - Preprocess PGN data (run once)

Parses PGN files and saves positions as binary chunks for fast training. The model is trained on the move actually played in the PGN. If Stockfish annotations such as `[%best_move: ...]` are present, preprocessing reports how often the played move matched the engine top move and stores that as metadata, but it does not use Stockfish's move as the training label.

By default, preprocessing reads `extractions/GM_games_2600.zip` and `extractions/magnus.zip`. Set `GM_ZIP` or `MAGNUS_ZIP` only if you want to use files somewhere else.

Preprocessing also stores `cp_loss`, `sample_weight`, `is_bad_move`, `cp_loss_bucket`, and `value_target` metadata for each position. `value_target` is a bounded side-to-move score, using Stockfish/eval annotations when available and the final game result as a fallback. `is_bad_move` is set from `BAD_MOVE_CP_LOSS_THRESHOLD`, and `cp_loss_bucket` marks unknown/ok/inaccuracy/mistake/blunder/critical categories. Set `STOCKFISH_PATH` if `stockfish.exe` is not in the project root. You can tune analysis with `CP_LOSS_TIME_LIMIT` or `CP_LOSS_DEPTH`, or set `CALCULATE_CP_LOSS=0` to skip the Stockfish metadata pass.

The board tensor has 17 perspective-relative channels: 6 own-piece planes, 6 opponent-piece planes, 4 perspective-relative castling-right planes, and 1 en-passant target plane. Chunks include a `board_encoding` marker so old absolute white/black chunks fail loudly instead of training silently wrong. The model predicts one fixed move-policy class for each `(from, to, promotion)` combination, then masks illegal moves during training and inference. Re-run preprocessing after architecture changes so saved chunks match the current model input shape and policy metadata.

Preprocessing writes separate train, validation, and test chunks. Defaults are `TRAIN_SPLIT=0.80`, `VAL_SPLIT=0.10`, and `TEST_SPLIT=0.10`. Use fresh output directories for each preprocessing run, for example:
```
$env:TRAIN_DIR = "data/train_chunks_perspective_v2"
$env:VAL_DIR = "data/val_chunks_perspective_v2"
$env:TEST_DIR = "data/test_chunks_perspective_v2"
python preprocess.py
```

The split is by game, not by individual position. When both the GM archive and Magnus archive are used, preprocessing skips Magnus games whose headers already appear in the GM archive so duplicate games do not leak across train/validation/test.

To add Lichess strong-player-vs-lower-rated games, first extract a smaller tagged PGN:
```
python extract_lichess_gm_vs_lower.py `
  --input "C:\Users\Vincent\Downloads\lichess_db_standard_rated_2026-04.pgn.zst" `
  --output "extractions\lichess_2500_vs_u2200_2026-04.pgn" `
  --gm_min_elo 2500 `
  --opponent_max_elo 2200 `
  --max_games 100000
```

Then preprocess that tagged PGN. Positive targets come only from `TrainingPolicyColor` / `StrongSide`; lower-rated moves with high CP loss are saved as negative targets so training can learn to avoid them:
```
$env:CP_LOSS_DEPTH = "6"
$env:BAD_MOVE_CP_LOSS_THRESHOLD = "150"
$env:INACCURACY_CP_LOSS_THRESHOLD = "50"
$env:MISTAKE_CP_LOSS_THRESHOLD = "150"
$env:BLUNDER_CP_LOSS_THRESHOLD = "300"
$env:CRITICAL_CP_LOSS_THRESHOLD = "900"
python preprocess.py `
  --single_pgn "extractions\lichess_2500_vs_u2200_2026-04.pgn" `
  --output_dir "data\train_chunks_lichess_2500_vs_u2200_v1" `
  --policy_color_mode tagged
```

To add shallow alpha-beta labels for bad candidate moves, enable
search-assisted negatives during preprocessing. This keeps the played strong
move as the positive label, then searches a few other legal moves and stores
only the clearly worse alternatives as negative targets:
```
python preprocess.py `
  --single_pgn "extractions\lichess_2500_vs_u2200_2026-04.pgn" `
  --output_dir "data\train_chunks_lichess_search_negatives_perspective_v2" `
  --policy_color_mode tagged `
  --search_negative_candidates 8 `
  --search_negative_max_per_position 2 `
  --search_negative_depth 2 `
  --search_negative_threshold 250 `
  --search_negative_only `
  --no_cp_loss
```

Update the paths in `preprocess.py` to match your machine, then run:
```
python preprocess.py
```

This takes 20-30 minutes and saves chunks to `data/train_chunks/`, `data/val_chunks/`, and `data/test_chunks/` by default.

### Step 2 - Train the model

First verify that the Python environment you are about to use can see CUDA:
```
$env:REQUIRE_CUDA = "1"
python check_training_env.py
```

```
$env:TRAIN_DIR = "data/train_chunks_perspective_v2"
$env:VAL_DIR = "data/val_chunks_perspective_v2"
$env:MODEL_PATH = "model/grandmaster_model_perspective_resnet_v2.pt"
$env:REQUIRE_CUDA = "1"
python neural_network.py
```

Training runs for up to 50 epochs with early stopping. On an RTX 3070, each epoch takes longer than the old two-layer CNN because the model now uses a padded residual tower. `REQUIRE_CUDA=1` makes training fail immediately instead of silently falling back to CPU. Set `TRAIN_LOG_INTERVAL` to control batch progress logging. Set `MODEL_PATH` to the checkpoint name you want; without it, training defaults to `model/grandmaster_model_perspective_resnet_negatives_v2.pt`. Avoid warm-starting from old absolute-channel checkpoints unless you are only doing a quick experiment; the board encoding and trunk architecture changed.

To train on the original GM/Magnus chunks plus the Lichess negative-example chunks, use `TRAIN_DIRS` separated by semicolons on Windows:
```
$env:TRAIN_DIRS = "data/train_chunks_perspective_v2;data/train_chunks_lichess_2500_vs_u2200_perspective_v2;data/train_chunks_lichess_search_negatives_perspective_v2"
$env:VAL_DIR = "data/val_chunks_perspective_v2"
$env:MODEL_PATH = "model/grandmaster_model_perspective_resnet_negatives_v2.pt"
python neural_network.py
```

### Step 3 - Test the model

```
python load_model.py
```

Loads the trained model and predicts the first move from the starting position.

For held-out test-set metrics and example predictions:
```
$env:TEST_DIR = "data/test_chunks_perspective_v2"
$env:MODEL_PATH = "model/grandmaster_model_perspective_resnet_negatives_v2.pt"
python evaluate_model.py --model $env:MODEL_PATH --examples 10
```

---

## Backend API

The web interface uses a Flask backend to run Alphabeta and Magnus Carlsen moves server-side. The implementation lives in `backend/app.py`; root `app.py` is a compatibility wrapper.

**Run locally:**
```
pip install flask flask-cors
$env:MODEL_PATH = "model/grandmaster_model_perspective_resnet_negatives_v2.pt"
$env:MAGNUS_TEMPERATURE = "0"
$env:MAGNUS_VALUE_WEIGHT = "2.0"
$env:MAGNUS_VALUE_CANDIDATES = "0"
python app.py
```

The server starts at `http://localhost:5000`. The web interface automatically connects to it when opened on localhost.

**API endpoints:**

`GET /` - health check, returns available players and the loaded Magnus model

`POST /api/move` - get the next move
```json
{ "fen": "<FEN string>", "player": "alphabeta", "depth": 3 }
{ "fen": "<FEN string>", "player": "magnus", "temperature": 0.0, "value_weight": 2.0, "value_candidates": 0 }
```

---

## Project structure

```
playable-chess-AI/
backend/                 Flask API package
frontend/src/api/        Browser API client
frontend/src/components/ DOM-facing UI controllers
frontend/src/game/       Chess game orchestration
frontend/src/services/   Browser engine wrappers
frontend/src/styles/     CSS
index.html               Frontend app shell for GitHub Pages
app.py                   Compatibility wrapper for backend.app
main.py                  Desktop chess GUI (Pygame)
chess_player.py          Player implementations
neural_network.py        Model architecture and training pipeline
load_model.py            Model loading and move prediction
preprocess.py            PGN -> .npz chunk conversion
heuristics.py            Piece-square tables and endgame evaluation
stockfish.js             Stockfish WebAssembly for browser play
pieces/                  Chess piece PNG images
model/                   Saved PyTorch checkpoints, ignored by Git
data/                    Generated training chunks, ignored by Git
```

---

## Deployment

The web backend is deployed to Railway. Pushing to the `main` branch triggers an automatic redeploy.

To deploy your own instance, connect your GitHub repo to Railway and set the start command to `python -m backend.app`. The backend URL lives in `frontend/src/config.js`:

```javascript
apiUrl: isFile || isLocalHost
  ? 'http://localhost:5000'
  : 'https://your-railway-url.up.railway.app',
```
