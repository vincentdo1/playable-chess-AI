#!/usr/bin/env bash
#
# Full training pipeline: GM supervised pretraining -> AlphaZero self-play.
#
# Stage 1  preprocess GM PGNs -> train/val/test .npz chunks
# Stage 2  supervised-train the CURRENT architecture on those chunks -> $MODEL_PATH
# Stage 3  self-play from $MODEL_PATH, improving policy+value via MCTS
#
# Why this exists: the repo ships a trained checkpoint (grandmaster_model_policy_v1.pt)
# that is policy-only and built for an older, smaller architecture. It does NOT fit
# the current ChessModel (128 filters / 8 blocks / no pooling) and has no value head,
# which MCTS needs. This script rebuilds a matching base model with both heads from
# the GM data, then self-plays from it.
#
# Run on a GPU box. Each stage is resumable: comment out earlier stages once done.
#
# Usage:
#   ./train_pipeline.sh                 # full run with defaults
#   STOCKFISH_PATH=/usr/bin/stockfish ./train_pipeline.sh   # better value targets
#   SKIP_PREPROCESS=1 ./train_pipeline.sh                   # reuse existing chunks
#
set -euo pipefail

# --- Shared config -----------------------------------------------------------
export TRAIN_DIR="${TRAIN_DIR:-data/train_chunks}"
export VAL_DIR="${VAL_DIR:-data/val_chunks}"
export TEST_DIR="${TEST_DIR:-data/test_chunks}"
# The current-architecture base model that self-play will bootstrap from.
export MODEL_PATH="${MODEL_PATH:-model/grandmaster_model_perspective_resnet_negatives_v2.pt}"

# Supervised training hyperparameters (env-read by neural_network.py).
export EPOCHS="${EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export RESIDUAL_FILTERS="${RESIDUAL_FILTERS:-128}"
export RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-8}"

PYTHON="${PYTHON:-python3}"

# --- Stage 1: preprocess -----------------------------------------------------
# If a real Stockfish binary is on STOCKFISH_PATH, value targets come from engine
# evals and bad-move negatives are mined. Otherwise pass --no_cp_loss to fall back
# to game-result value targets (which happen to match self-play's value convention).
if [ "${SKIP_PREPROCESS:-0}" != "1" ]; then
  echo "=== Stage 1/3: preprocessing GM PGNs -> chunks ==="
  PREPROCESS_FLAGS="${PREPROCESS_FLAGS:-}"
  if ! command -v "${STOCKFISH_PATH:-stockfish}" >/dev/null 2>&1 && [ ! -x "${STOCKFISH_PATH:-}" ]; then
    echo "  (no Stockfish found -> --no_cp_loss, value targets from game result)"
    PREPROCESS_FLAGS="$PREPROCESS_FLAGS --no_cp_loss"
  fi
  $PYTHON preprocess.py $PREPROCESS_FLAGS
else
  echo "=== Stage 1/3: SKIPPED (SKIP_PREPROCESS=1) ==="
fi

# --- Stage 2: supervised pretraining ----------------------------------------
if [ "${SKIP_SUPERVISED:-0}" != "1" ]; then
  echo "=== Stage 2/3: supervised pretraining current architecture -> $MODEL_PATH ==="
  $PYTHON neural_network.py
else
  echo "=== Stage 2/3: SKIPPED (SKIP_SUPERVISED=1) ==="
fi

# --- Stage 3: self-play ------------------------------------------------------
echo "=== Stage 3/3: AlphaZero self-play from $MODEL_PATH ==="
$PYTHON train_self_play.py \
  --init_checkpoint "$MODEL_PATH" \
  --iterations "${SP_ITERATIONS:-20}" \
  --games_per_iteration "${SP_GAMES:-50}" \
  --training_steps "${SP_TRAIN_STEPS:-1000}" \
  --mcts_simulations "${SP_SIMS:-400}" \
  --mcts_batch_size "${SP_BATCH:-16}" \
  --batch_size "${SP_TRAIN_BATCH:-256}"

echo "=== Pipeline complete. Self-play checkpoints in model/selfplay_checkpoints/ ==="
