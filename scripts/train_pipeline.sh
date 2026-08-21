#!/usr/bin/env bash
# GM preprocessing, supervised training, and AlphaZero self-play.
# Defaults use the v3 encoding and model paths. Override ARCH_VERSION and the
# data/model environment variables for legacy training.
set -euo pipefail

# Configuration
export TRAIN_DIR="${TRAIN_DIR:-data/train_chunks_v3}"
export VAL_DIR="${VAL_DIR:-data/val_chunks_v3}"
export TEST_DIR="${TEST_DIR:-data/test_chunks_v3}"
export MODEL_PATH="${MODEL_PATH:-model/grandmaster_resnet_v3.pt}"

export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export RESIDUAL_FILTERS="${RESIDUAL_FILTERS:-128}"
export RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-8}"
export TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-100}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Disable only for a bounded CPU smoke test.
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

if [ -z "${PYTHON:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "No Python executable found. Set PYTHON=/path/to/python and rerun."
    exit 1
  fi
fi

if [ "${SKIP_ENV_CHECK:-0}" != "1" ]; then
  $PYTHON -m training.check_training_env
fi

# Preprocess
if [ "${SKIP_PREPROCESS:-0}" != "1" ]; then
  echo "=== Stage 1/3: preprocessing GM PGNs -> chunks ==="
  PREPROCESS_FLAGS="${PREPROCESS_FLAGS:-}"
  if ! command -v "${STOCKFISH_PATH:-stockfish}" >/dev/null 2>&1 && [ ! -x "${STOCKFISH_PATH:-}" ]; then
    echo "  (no Stockfish found -> --no_cp_loss, value targets from game result)"
    PREPROCESS_FLAGS="$PREPROCESS_FLAGS --no_cp_loss"
  fi
  $PYTHON -m training.preprocess $PREPROCESS_FLAGS
else
  echo "=== Stage 1/3: SKIPPED (SKIP_PREPROCESS=1) ==="
fi

# Supervised pretraining
if [ "${SKIP_SUPERVISED:-0}" != "1" ]; then
  echo "=== Stage 2/3: supervised pretraining current architecture -> $MODEL_PATH ==="
  $PYTHON neural_network.py
else
  echo "=== Stage 2/3: SKIPPED (SKIP_SUPERVISED=1) ==="
fi

# Self-play
echo "=== Stage 3/3: AlphaZero self-play from $MODEL_PATH ==="
$PYTHON -m experiments.train_self_play \
  --init_checkpoint "$MODEL_PATH" \
  --iterations "${SP_ITERATIONS:-20}" \
  --games_per_iteration "${SP_GAMES:-200}" \
  --training_steps "${SP_TRAIN_STEPS:-100}" \
  --mcts_simulations "${SP_SIMS:-400}" \
  --mcts_batch_size "${SP_BATCH:-16}" \
  --batch_size "${SP_TRAIN_BATCH:-256}"

echo "=== Pipeline complete. Self-play checkpoints in model/selfplay_checkpoints/ ==="
