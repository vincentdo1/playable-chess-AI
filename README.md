# Chess AI — Vincent Do

A playable chess application featuring multiple AI players including a neural network trained on Magnus Carlsen's games. Available as a desktop app and a web interface.

---

## Project overview

This project combines several approaches to chess AI:

- **Random** — picks a legal move at random
- **Alphabeta** — minimax search with alpha-beta pruning, endgame-aware heuristics, and move ordering
- **Stockfish** — UCI engine integration (desktop only)
- **Magnus Carlsen NN** — a CNN + LSTM neural network trained on Magnus Carlsen's annotated games, with hybrid alphabeta search for tactical correctness

---

## Web interface

The easiest way to play. No setup required.

**Live site:** `https://vincentdo1.github.io/playable-chess-AI`

Features available in the browser:
- Human, Stockfish (WebAssembly), Random, Alphabeta AI, and Magnus Carlsen NN
- Adjustable Stockfish skill level (0–20)
- Adjustable alphabeta search depth
- Move history, status display, flip board, undo move

The web interface calls a Flask backend hosted on Railway for the Alphabeta and Magnus Carlsen players. Stockfish and Random run entirely in the browser with no server needed.

---

## Desktop app

### Requirements

- Python 3.12
- An Nvidia GPU (recommended for Magnus Carlsen NN inference)
- Stockfish 17 — download from `https://stockfishchess.org/download/`

### Installation

**Step 1 — Create a virtual environment with Python 3.12:**
```
py -3.12 -m venv chess_env
chess_env\Scripts\activate
```

**Step 2 — Install PyTorch with CUDA support:**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Step 3 — Install remaining dependencies:**
```
pip install -r requirements-local.txt
```

**Step 4 — Add Stockfish**

Place `stockfish.exe` in the project root, or set the environment variable to point to it anywhere on your machine:
```
set STOCKFISH_PATH=C:\path\to\stockfish.exe
```

**Step 5 — Add the trained model**

Place `grandmaster_model_v2.pt` in the `data/` folder:
```
data/grandmaster_model_v2.pt
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

### Step 1 — Preprocess PGN data (run once)

Parses annotated PGN files and saves positions as binary chunks for fast training. Requires `magnus.zip` and `GM_games_2600.zip` in the `extractions/` folder.

Update the paths in `preprocess.py` to match your machine, then run:
```
python preprocess.py
```

This takes 20–30 minutes and saves chunks to `data/train_chunks/` and `data/val_chunks/`.

### Step 2 — Train the model

```
python neural_network.py
```

Training runs for up to 50 epochs with early stopping. On an RTX 3070, each epoch takes approximately 15–25 minutes. The best model is saved automatically to `data/grandmaster_model_v2.pt`.

### Step 3 — Test the model

```
python load_model.py
```

Loads the trained model and predicts the first move from the starting position.

---

## Flask backend (for web interface)

The web interface uses a Flask backend to run Alphabeta and Magnus Carlsen moves server-side.

**Run locally:**
```
pip install flask flask-cors
python app.py
```

The server starts at `http://localhost:5000`. The web interface automatically connects to it when opened on localhost.

**API endpoints:**

`GET /` — health check, returns available players

`POST /api/move` — get the next move
```json
{ "fen": "<FEN string>", "player": "alphabeta", "depth": 3 }
{ "fen": "<FEN string>", "player": "magnus" }
```

---

## Project structure

```
playable-chess-AI/
├── main.py              — desktop chess GUI (Pygame)
├── chess_player.py      — player implementations (random, alphabeta, stockfish)
├── neural_network.py    — model architecture and training pipeline
├── load_model.py        — model loading and move prediction
├── preprocess.py        — PGN → .npz chunk conversion
├── heuristics.py        — piece-square tables and endgame evaluation
├── pgn_parser.py        — PGN annotation with Stockfish evaluations
├── app.py               — Flask backend for web interface
├── index.html           — web interface (GitHub Pages)
├── stockfish.js         — Stockfish WebAssembly (for browser play)
├── pieces/              — chess piece PNG images
├── extractions/         — raw PGN zip files (not in Git)
└── data/                — training chunks and saved model (not in Git)
```

---

## Deployment

The web backend is deployed to Railway. Pushing to the `main` branch triggers an automatic redeploy.

To deploy your own instance, connect your GitHub repo to Railway and set the start command to `python app.py`. The backend URL must be updated in `index.html`:

```javascript
var API_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000'
  : 'https://your-railway-url.up.railway.app';
```