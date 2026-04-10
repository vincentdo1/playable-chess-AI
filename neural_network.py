"""
neural_network.py — Chess move prediction model (optimized for training speed).

Workflow:
  1. Make sure you've run preprocess.py to convert PGN data into .npz chunks.
  2. Run this file to train:  python neural_network.py
"""

import os
import glob
import chess.pgn
import chess
import re
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, LSTM, Dropout
from keras.models import Model

import numpy as np

# ---------------------------------------------------------------------------
# Config — edit paths to match your setup
# ---------------------------------------------------------------------------
TRAIN_DIR  = 'data/train_chunks'
VAL_DIR    = 'data/val_chunks'
MODEL_PATH = 'data/grandmaster_model_v2.keras'
BATCH_SIZE = 256    # increase if you have a large GPU; decrease if you run OOM
EPOCHS     = 50     # EarlyStopping will halt before this if val loss plateaus
SEQ_LEN    = 10
# ---------------------------------------------------------------------------

piece_to_index = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
    'ck': 12, 'cq': 13, 'CK': 14, 'CQ': 15
}


def fen_to_tensor(fen):
    """Convert a FEN string to an 8x8x16 numpy tensor."""
    tensor = np.zeros((8, 8, 16), dtype=np.float32)
    parts = fen.split(' ')
    rows = parts[0].split('/')
    for i, row in enumerate(rows):
        tensor_row = 7 - i
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                tensor[tensor_row, j, piece_to_index[char]] = 1
                j += 1
    castling = parts[2]
    if 'K' in castling:
        tensor[:, :, piece_to_index['CK']] = 1
    if 'Q' in castling:
        tensor[:, :, piece_to_index['CQ']] = 1
    if 'k' in castling:
        tensor[:, :, piece_to_index['ck']] = 1
    if 'q' in castling:
        tensor[:, :, piece_to_index['cq']] = 1
    return tensor


def square_to_index(square):
    """Convert an algebraic square name (e.g. 'e4') to a python-chess index (0-63, a1=0)."""
    return chess.parse_square(square.lower()[:2])


def move_to_vector(move):
    """Encode a chess.Move as a 132-dim float vector for use as LSTM input."""
    vector = np.zeros(132, dtype=np.float32)
    vector[move.from_square] = 1
    vector[64 + move.to_square] = 1
    if move.promotion is not None:
        promo_map = {chess.KNIGHT: 128, chess.BISHOP: 129,
                     chess.ROOK: 130, chess.QUEEN: 131}
        idx = promo_map.get(move.promotion)
        if idx is not None:
            vector[idx] = 1
    return vector


def move_sequence_to_vector(move_sequence, max_length=10):
    """
    Encode a sequence of chess.Move objects as a (max_length, 132) float matrix.
    Each row is a move encoded the same way as move_to_vector.
    """
    sequence_vector = np.zeros((max_length, 132), dtype=np.float32)
    for i, move in enumerate(move_sequence[-max_length:]):
        move_vector = np.zeros(132, dtype=np.float32)
        move_vector[move.from_square] = 1
        move_vector[64 + move.to_square] = 1
        if move.promotion is not None:
            promo_map = {chess.KNIGHT: 128, chess.BISHOP: 129,
                         chess.ROOK: 130, chess.QUEEN: 131}
            idx = promo_map.get(move.promotion)
            if idx is not None:
                move_vector[idx] = 1
        sequence_vector[i] = move_vector
    return sequence_vector


def move_to_index(move, board):
    """
    Return (from_square_index, to_square_index) as two separate integers (0-63).
    Used by preprocess.py to compute target labels.
    """
    if move not in board.legal_moves:
        raise ValueError(f"Move {move.uci()} is not legal in this position.")
    return move.from_square, move.to_square


def parse_pgn(file_path, sequence_length=10):
    """
    Generator over annotated moves in a PGN file.
    Used by preprocess.py — NOT called during training in the optimized pipeline.
    """
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            recent_moves = []
            for node in game.mainline():
                move = node.move
                fen = board.fen()
                board_tensor = fen_to_tensor(fen)
                recent_moves.append(move)
                if len(recent_moves) > sequence_length:
                    recent_moves.pop(0)
                move_vector = move_to_vector(move)
                comment = node.comment
                eval_match          = re.search(r'\[%eval: ([^\]]+)\]', comment)
                best_move_match     = re.search(r'\[%best_move: ([^\]]+)\]', comment)
                played_best_move_match = re.search(r'\[%played_best_move: ([^\]]+)\]', comment)
                evaluation       = eval_match.group(1)          if eval_match          else None
                best_move        = best_move_match.group(1)     if best_move_match     else None
                played_best_move = played_best_move_match.group(1) if played_best_move_match else None
                if best_move is None:
                    board.push(move)
                    continue
                from_idx, to_idx = move_to_index(board.parse_uci(best_move), board)
                yield (board_tensor, recent_moves, move_vector,
                       evaluation, best_move, played_best_move, (from_idx, to_idx))
                board.push(move)


# ---------------------------------------------------------------------------
# OPTIMIZATION A — tf.data pipeline
# ---------------------------------------------------------------------------

def _load_chunk(path: str):
    """Load one .npz chunk and return its four arrays."""
    data = np.load(path)
    return data['boards'], data['moves'], data['from_sq'], data['to_sq']


def create_dataset(chunk_dir: str, batch_size: int = BATCH_SIZE,
                   shuffle: bool = True) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset that streams from pre-saved .npz chunk files.

    OPTIMIZATION A: Using tf.data with prefetch overlaps data loading on the
    CPU with model execution on the GPU. Without this, the GPU sits idle for
    every batch while the CPU finishes loading the next one.

    OPTIMIZATION E: No PGN parsing — data was already converted by preprocess.py.
    This eliminates the #1 bottleneck (re-parsing 300MB of PGN per epoch).
    """
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk files found in {chunk_dir!r}. "
            "Run preprocess.py first."
        )

    def generator():
        paths = chunk_paths.copy()
        if shuffle:
            import random
            random.shuffle(paths)
        for path in paths:
            boards, moves, from_sq, to_sq = _load_chunk(path)
            # Shuffle within each chunk
            if shuffle:
                idx = np.random.permutation(len(boards))
                boards, moves, from_sq, to_sq = (
                    boards[idx], moves[idx], from_sq[idx], to_sq[idx]
                )
            for i in range(len(boards)):
                yield (
                    {'board_input': boards[i], 'move_input': moves[i]},
                    {'from_output': from_sq[i], 'to_output': to_sq[i]}
                )

    output_signature = (
        {
            'board_input': tf.TensorSpec(shape=(8, 8, 16), dtype=tf.float32),
            'move_input':  tf.TensorSpec(shape=(10, 132),  dtype=tf.float32),
        },
        {
            'from_output': tf.TensorSpec(shape=(), dtype=tf.int32),
            'to_output':   tf.TensorSpec(shape=(), dtype=tf.int32),
        }
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)  # overlap CPU loading with GPU training
    return ds


# ---------------------------------------------------------------------------
# OPTIMIZATION B — Mixed precision
# ---------------------------------------------------------------------------

def enable_mixed_precision():
    """
    OPTIMIZATION B: Mixed precision runs most operations in float16 while
    keeping a float32 master copy of weights for numerical stability.
    This gives ~1.5-2x speedup on Nvidia GPUs (Turing/Ampere and newer)
    with no meaningful loss in model quality.

    If you hit NaN losses after enabling this, disable it — it means your
    GPU doesn't support it well or your gradients are unstable.
    """
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (float16 compute, float32 weights).")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_chess_model():
    """
    Build the chess move prediction model with two output heads.

    Architecture is unchanged from the bug-fixed version. The two 64-class
    softmax heads (from_output, to_output) trained with sparse_categorical_
    crossentropy are the correct formulation for move prediction.

    OPTIMIZATION B: mixed precision is enabled globally before calling this,
    so the model automatically uses float16 compute where beneficial.
    """
    board_input = Input(shape=(8, 8, 16), name='board_input')
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(board_input)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)

    move_input = Input(shape=(10, 132), name='move_input')
    y = LSTM(64)(move_input)

    combined = tf.keras.layers.concatenate([x, y])
    z = Dense(256, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)

    # Cast back to float32 before softmax — required when using mixed precision,
    # because float16 softmax can produce inf/NaN near the output boundaries.
    from_logits = Dense(64, name='from_logits')(z)
    to_logits   = Dense(64, name='to_logits')(z)
    from_output = tf.keras.layers.Activation('softmax', dtype='float32', name='from_output')(from_logits)
    to_output   = tf.keras.layers.Activation('softmax', dtype='float32', name='to_output')(to_logits)

    model = Model(inputs=[board_input, move_input], outputs=[from_output, to_output])

    # OPTIMIZATION C: larger batch size (set via BATCH_SIZE constant) and a
    # slightly higher initial learning rate to compensate — large batches need
    # a proportionally larger lr to converge at the same rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss={
            'from_output': 'sparse_categorical_crossentropy',
            'to_output':   'sparse_categorical_crossentropy',
        },
        metrics={
            'from_output': 'accuracy',
            'to_output':   'accuracy',
        }
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_callbacks(model_path: str = MODEL_PATH):
    """
    OPTIMIZATION D — training callbacks.

    ModelCheckpoint   : saves the model only when val loss improves.
                        You always keep the best version, not just the last.
    EarlyStopping     : stops training if val loss hasn't improved for 5
                        epochs, saving time and preventing overfitting.
    ReduceLROnPlateau : halves the learning rate when val loss stalls for
                        3 epochs, helping the model escape plateaus without
                        requiring you to manually tune the schedule.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath        = model_path,
            monitor         = 'val_loss',
            save_best_only  = True,
            verbose         = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
    ]


def main():
    # OPTIMIZATION B: enable before building the model
    enable_mixed_precision()

    model = create_chess_model()
    model.summary()

    print(f"\nLoading training data from   : {TRAIN_DIR}/")
    print(f"Loading validation data from : {VAL_DIR}/")
    print(f"Batch size : {BATCH_SIZE}  |  Max epochs : {EPOCHS}\n")

    # OPTIMIZATION A + E: tf.data pipeline over pre-processed chunks
    train_ds = create_dataset(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = create_dataset(VAL_DIR,   batch_size=BATCH_SIZE, shuffle=False)

    # OPTIMIZATION D: model.fit() is more optimized than a manual train_on_batch
    # loop and integrates cleanly with callbacks and tf.data
    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = build_callbacks(),
    )

    print("\nTraining complete. Best model saved to:", MODEL_PATH)


if __name__ == '__main__':
    main()