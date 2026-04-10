from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
import numpy as np
from neural_network import fen_to_tensor, move_sequence_to_vector
import chess

# Must match the policy set in neural_network.py before the model was built and
# saved. Without this, Keras cannot correctly restore the float16 layer dtypes
# from the saved config, which can cause dtype mismatches at inference time.
mixed_precision.set_global_policy('mixed_float16')

print("Loading model...")
model = load_model('data/grandmaster_model_v2.keras')


def predict_next_move(model, board):
    """
    Predict the next move for the given board position.

    Returns the predicted move as a UCI string (e.g. 'e2e4'), or None if the
    board has no legal moves.
    """
    board_tensor = fen_to_tensor(board.fen())

    # Pass chess.Move objects — move_sequence_to_vector uses .from_square / .to_square
    move_sequence_vector = move_sequence_to_vector(
        list(board.move_stack[-10:]),
        max_length=10
    )

    # Model returns two outputs: (from_probs, to_probs), each shape (1, 64)
    from_probs, to_probs = model.predict(
        [np.array([board_tensor]), np.array([move_sequence_vector])],
        verbose=0
    )

    # Rank each square independently by probability, then find the first legal
    # combination. Top-8 candidates cover the vast majority of legal moves.
    from_candidates = np.argsort(from_probs[0])[::-1][:8]
    to_candidates   = np.argsort(to_probs[0])[::-1][:8]

    for from_sq in from_candidates:
        for to_sq in to_candidates:
            move = chess.Move(from_sq, to_sq)
            if move in board.legal_moves:
                return move.uci()

            # Also try queen promotion if the move lands on the back rank
            promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            if promo in board.legal_moves:
                return promo.uci()

    # Fallback: guarantees a legal move is always returned
    legal = list(board.legal_moves)
    if legal:
        print("Warning: model produced no legal move -- falling back to first legal move.")
        return legal[0].uci()
    return None


if __name__ == '__main__':
    board = chess.Board()
    next_move = predict_next_move(model, board)
    print("Predicted next move:", next_move)