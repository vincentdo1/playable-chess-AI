"""
load_model.py — Load a trained ChessModel and predict the next move.
"""

import numpy as np
import torch
import chess

from neural_network import (ChessModel, fen_to_tensor, move_sequence_to_vector,
                             flip_square, MODEL_PATH, DEVICE)


def load_trained_model(path: str = MODEL_PATH) -> ChessModel:
    checkpoint = torch.load(path, map_location=DEVICE)
    model = ChessModel().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {path}")
    print(f"  Saved at epoch {checkpoint['epoch']}  |  val_loss={checkpoint['val_loss']:.4f}")
    return model


def predict_next_move(model: ChessModel, board: chess.Board,
                      temperature: float = 1.2) -> str | None:
    """
    Predict the next move for the given board position.

    The board is flipped when it is Black's turn so the model always sees
    the position from the perspective of the side to move — matching the
    representation used during training.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    is_black = (board.turn == chess.BLACK)

    board_tensor = fen_to_tensor(board.fen(), flip=is_black)
    move_seq     = move_sequence_to_vector(list(board.move_stack[-10:]),
                                           max_length=10, flip=is_black)

    board_t = torch.tensor(board_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    move_t  = torch.tensor(move_seq,    dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        from_logits, to_logits = model(board_t, move_t)

    from_probs = torch.softmax(from_logits[0], dim=0).cpu().numpy()
    to_probs   = torch.softmax(to_logits[0],   dim=0).cpu().numpy()

    # Score every legal move. When Black, flip the move squares to match
    # the flipped board representation the model was trained and inferred on,
    # then convert back to real squares when returning the move.
    scores = []
    for move in legal_moves:
        from_sq = flip_square(move.from_square) if is_black else move.from_square
        to_sq   = flip_square(move.to_square)   if is_black else move.to_square
        scores.append(float(from_probs[from_sq]) * float(to_probs[to_sq]))

    scores = np.array(scores)

    if temperature == 0.0:
        return legal_moves[int(np.argmax(scores))].uci()

    scores = np.power(scores, 1.0 / temperature)
    total  = scores.sum()
    if total == 0:
        return np.random.choice(legal_moves).uci()

    probs  = scores / total
    chosen = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[chosen].uci()


if __name__ == '__main__':
    print("Loading model...")
    model = load_trained_model()
    board = chess.Board()
    move  = predict_next_move(model, board)
    print(f"Predicted first move: {move}")