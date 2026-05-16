"""Load a trained ChessModel and predict the next move."""

import numpy as np
import torch
import chess

from neural_network import (ChessModel, fen_to_tensor, move_sequence_to_vector,
                             move_to_policy_index, MODEL_PATH, DEVICE)

def load_trained_model(path: str = MODEL_PATH) -> ChessModel:
    """Load a saved ChessModel from a .pt checkpoint file."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model = ChessModel().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {path}")
    print(f"  Saved at epoch {checkpoint['epoch']}  |  val_loss={checkpoint['val_loss']:.4f}")
    return model

def _get_move_scores(model: ChessModel, board: chess.Board):
    """
    Run the neural network and return a score for every legal move.

    Returns a list of (score, move) tuples sorted highest score first.
    Scores are fixed move-policy logits, with board flipping applied for Black.
    """
    is_black = (board.turn == chess.BLACK)

    board_tensor = fen_to_tensor(board.fen(), flip=is_black)
    move_seq     = move_sequence_to_vector(list(board.move_stack[-10:]),
                                           max_length=10, flip=is_black)
    model_device = next(model.parameters()).device

    board_t = torch.tensor(board_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(model_device)
    move_t  = torch.tensor(move_seq,    dtype=torch.float32).unsqueeze(0).to(model_device)

    with torch.no_grad():
        policy_logits = model(board_t, move_t)

    scored = []
    for move in board.legal_moves:
        move_idx = move_to_policy_index(move, flip=is_black)
        scored.append((policy_logits[0, move_idx].item(), move))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def predict_next_move(model: ChessModel, board: chess.Board,
                      temperature: float = 1.2) -> str | None:
    """
    Pure neural network prediction with temperature sampling.
    Fast — no search. Best for opening moves where the model is confident.

    temperature controls variety:
      0.0 = always pick the single best move (greedy)
      1.2 = natural variety while still preferring confident moves
      2.0 = very exploratory
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    scored = _get_move_scores(model, board)
    log_scores = np.array([s for s, _ in scored])
    moves  = [m for _, m in scored]

    if temperature == 0.0:
        return moves[0].uci()

    log_scores = log_scores / temperature
    log_scores -= log_scores.max()   # subtract max for stability
    probs = np.exp(log_scores)
    probs /= probs.sum()

    chosen = np.random.choice(len(moves), p=probs)
    return moves[chosen].uci()

if __name__ == '__main__':
    print("Loading model...")
    model = load_trained_model()

    board = chess.Board()

    print("\n--- Pure NN prediction (deterministic) ---")
    move = predict_next_move(model, board, temperature=0.0)
    print(f"Predicted first move: {move}")

    print("\n--- Pure NN prediction (sampled) ---")
    move = predict_next_move(model, board, temperature=1.2)
    print(f"Predicted first move: {move}")
