# Chess AI - Vincent Do

A playable chess app with random, alpha-beta, Stockfish, and neural-network
opponents. It runs in the browser with a Flask backend or as a Pygame desktop
app.

- [Live frontend](https://vincentdo1.github.io/playable-chess-AI)
- [v4 model notes](docs/MODEL_CARD.md)
- [Railway deployment](docs/RAILWAY_DEPLOY.md)

<p align="center">
  <img src="media/network.gif" alt="CNN and LSTM forward pass" width="720">
  <br>
  <em>Visualization of the legacy v2 network. <a href="media/network.mp4">MP4 version</a>.</em>
</p>

## Players

- **Random** chooses a legal move at random.
- **Alphabeta** uses minimax, alpha-beta pruning, and hand-written heuristics.
- **Stockfish** uses the desktop UCI engine or browser WebAssembly build.
- **Neural Network** uses a policy/value ResNet, optionally with MCTS.

The `magnus` API value and `magnus_carlsen` desktop option are legacy
identifiers for the neural-network player. They are not claims of affiliation
with Magnus Carlsen.

## Local setup

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For GPU training, install the appropriate PyTorch build from the
[official selector](https://pytorch.org/get-started/locally/). Then install the
remaining requirements:

```powershell
pip install -r requirements-local.txt
```

Model checkpoints are not stored in Git. Put a compatible checkpoint in
`model/`; the default path is `model/grandmaster_resnet_v3.pt`. Set
`MODEL_PATH` to use a different file.

Start the backend:

```powershell
$env:MAGNUS_USE_MCTS = "1"
$env:MAGNUS_MCTS_SIMULATIONS = "200"
python app.py
```

Serve the frontend from a second terminal:

```powershell
python -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000). If the neural-network
option is disabled, check the backend output for the model-loading error.

For the Pygame app:

```powershell
python main.py --black_player magnus_carlsen
```

Desktop player values are `you`, `random`, `alphabeta`, `engine`, and
`magnus_carlsen`.

The desktop `engine` player requires a Stockfish executable. Set
`STOCKFISH_PATH` when it is not available as `stockfish.exe` in the repo root.

## Repository layout

```text
backend/       Flask API
frontend/      Static browser app
inference/     MCTS, neural alpha-beta search, and blunder guard
training/      Data preparation and model training
evaluation/    Test-set metrics and match harnesses
experiments/   Experimental self-play
tests/         Unit and artifact-dependent checks
scripts/       Training, comparison, and visualization helpers
pieces/        Board and piece images
```

Important entry points:

| File | Purpose |
| --- | --- |
| `app.py` | Local Flask launcher |
| `main.py` | Pygame app |
| `neural_network.py` | v2/v3 models and supervised trainer |
| `load_model.py` | Checkpoint loading and policy inference |
| `backend/app.py` | API and production server target |
| `training/train_distill.py` | v4 engine-distillation trainer |

Run package scripts with `python -m`, for example
`python -m evaluation.eval_arena`.

## Models

| Version | Training signal | Architecture | Input |
| --- | --- | --- | --- |
| v2 | Human games | ResNet + move-history LSTM | 17-plane `perspective_v2` |
| v3 | Human games | 8-block, 128-filter ResNet | 20-plane `perspective_v3` |
| v4 | Lichess Stockfish evaluations | 12-block, 256-filter SE-ResNet | 20 planes; channels 17–19 masked |

v3 is the default local model. v4 shares v3's encoding label, so its checkpoints
must include `arch_version='v4'`. Because the v4 source data contains
four-field FENs, it did not supervise halfmove-clock or repetition inputs; the
model zeros those channels at inference.

The repository does not include checkpoints or enough information to reproduce
the historical v4 training run exactly. See the [model card](docs/MODEL_CARD.md)
for the recorded result and its limitations.

## Training

### v3 supervised path

The preprocessor reads the paths in `GM_ZIP` and `MAGNUS_ZIP`, which default
to `extractions/GM_games_2600.zip` and `extractions/magnus.zip`. Those
archives are not included.

```powershell
python -m training.preprocess
python neural_network.py
```

The defaults write chunks under `data/` and the checkpoint to
`model/grandmaster_resnet_v3.pt`. Set `ARCH_VERSION=v2` for the legacy
model, or use `TRAIN_DIR`, `VAL_DIR`, and `MODEL_PATH` to override paths.

To resume a v2/v3 run:

```powershell
python -m training.resume_training `
  --init model\source.pt `
  --output model\resumed.pt `
  --epochs 40
```

### v4 engine distillation

The v4 pipeline downloads Lichess evaluation shards and can take days on a GPU.
It requires a pinned Hugging Face dataset commit:

```powershell
$sourceRevision = "<40-to-64-character-Hugging-Face-commit>"
.\run_phase2.ps1 -Execute -SourceRevision $sourceRevision
```

To run the stages separately:

```powershell
python -m training.ingest_lichess_evals --num_shards 3 `
  --source_revision $sourceRevision
python -m training.train_distill
python -m training.train_distill --resume model\grandmaster_resnet_v4_distill.pt
```

New ingests write `ingest_manifest.json`, and the trainer verifies it before
training. Keep that manifest with the resulting checkpoint.

### Self-play

Self-play is experimental:

```powershell
python -m experiments.train_self_play `
  --init_checkpoint model\grandmaster_resnet_v3.pt `
  --iterations 20 `
  --mcts_simulations 400
```

Evaluate a self-play checkpoint against its starting model before using it.

## Evaluation

Head-to-head checkpoint comparison:

```powershell
python -m evaluation.eval_arena `
  --model_a model\candidate.pt `
  --model_b model\baseline.pt `
  --method_a mcts --method_b mcts `
  --paired --games 128 --sims 200
```

Held-out test metrics:

```powershell
python -m evaluation.evaluate_model --examples 10
```

Stockfish match:

```powershell
python -m evaluation.vs_stockfish `
  --model model\grandmaster_resnet_v4_full.pt `
  --mode mcts --sims 200 --mcts_batch_size 16 `
  --uci_elo 2500 --movetime 0.6 `
  --output_dir evaluation/results
```

Stockfish `UCI_Elo` configures a limited-strength opponent; it does not give
the model a human or FIDE rating. Keep the checkpoint, engine version, search
settings, openings, and hardware with any published result.

## Backend API

- `GET /livez` checks process liveness.
- `GET /readyz` checks whether required model capabilities loaded.
- `POST /api/move` returns a move for a FEN and player.

Examples:

```json
{ "fen": "<FEN>", "player": "alphabeta", "depth": 3 }
{ "fen": "<FEN>", "player": "magnus", "temperature": 0.0 }
{ "fen": "<FEN>", "player": "magnus", "use_mcts": true }
```

Common server variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `model/grandmaster_resnet_v3.pt` | Checkpoint path |
| `MAGNUS_USE_MCTS` | `0` | Enable MCTS |
| `MAGNUS_MCTS_SIMULATIONS` | `200` | Search budget |
| `MAGNUS_MCTS_MAX_SIMULATIONS` | `800` | Server-side ceiling |
| `MAGNUS_MCTS_TIME_LIMIT` | `0` | Search time limit in seconds |
| `MAGNUS_HF_REPO` | unset | Hugging Face model repository |
| `MAGNUS_HF_REVISION` | required with repo | Pinned model revision |
| `MAGNUS_MODEL_SHA256` | unset | Expected checkpoint checksum |
| `MAGNUS_ALLOWED_ORIGINS` | production and local origins | CORS allowlist |

Clients cannot enable MCTS when the server disables it or raise the search
budget above the server limit.

## Deployment

The frontend calls the Railway backend configured in
`frontend/src/config.js`. Checkpoints are gitignored, so deploying the code
alone does not deploy a model. Follow [docs/RAILWAY_DEPLOY.md](docs/RAILWAY_DEPLOY.md)
to configure a pinned model revision, checksum, readiness checks, and rollback.

## Tests

```bash
python -m pip install --requirement requirements-test.txt
python -m pytest -q
```

Tests that need a local checkpoint, dataset, or Stockfish binary may skip in a
clean clone. Run those tests after supplying the required files.

## License and third-party files

This repository does not currently include a license. Third-party components
and known provenance gaps are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
