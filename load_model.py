"""
load_model.py — Load a trained ChessModel and predict the next move.

Two prediction modes:
  predict_next_move            — pure neural network (fast, Magnus-style)
  predict_next_move_with_search — optional NN candidates + alphabeta search benchmark
"""

import numpy as np
import torch
import chess

from neural_network import (ChessModel, fen_to_tensor, move_sequence_to_vector,
                             move_to_policy_index, MODEL_PATH, DEVICE)
import heuristics
import chess_player

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

def predict_next_move_with_search(model: ChessModel, board: chess.Board,
                                  top_n: int = 5,
                                  depth: int = 2) -> str | None:
    """
    Hybrid: NN picks top_n candidates, alphabeta searches them for the best.
    Preserves Magnus style while avoiding tactical blunders.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Step 1 — get top_n NN candidates
    scored = _get_move_scores(model, board)
    candidates = [move for _, move in scored[:top_n]]

    # If only one legal move, return it immediately
    if len(candidates) == 1:
        return candidates[0].uci()

    # Step 2 — alphabeta search over candidates
    best_move  = candidates[0]
    best_eval  = float('-inf')
    alpha = float('-inf') 

    for move in candidates:
        board.push(move)
        eval_score = -_alphabeta_search(board, depth - 1, -float('inf'), -alpha)  # pass alpha
        board.pop()
        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move
            alpha = best_eval

    return best_move.uci()

def _evaluate_board(board: chess.Board) -> float:
    """Score from the perspective of the side to move (positive = good for me)."""
    raw = heuristics.evaluate(chess_player.evaluate_helper(board), board)
    return raw if board.turn == chess.WHITE else -raw

def _alphabeta_search(board: chess.Board, depth: int,
                      alpha: float, beta: float) -> float:
    """
    Standard alphabeta search returning score from the perspective of
    the side to move at this node. Used internally by
    predict_next_move_with_search.
    """
    if board.is_checkmate():
        return -10000 + depth
    if board.is_repetition(3):
        return 0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return _evaluate_board(board)

    best = float('-inf')
    for move in chess_player._order_moves(board):
        board.push(move)
        score = -_alphabeta_search(board, depth - 1, -beta, -alpha)
        board.pop()
        best  = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best

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
