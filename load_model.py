"""
load_model.py — Load a trained ChessModel and predict the next move.

Two prediction modes:
  predict_next_move            — pure neural network (fast, Magnus-style)
  predict_next_move_with_search — NN candidates + alphabeta search (stronger,
                                   especially in middlegame and endgame)
"""

import math
import numpy as np
import torch
import chess

from neural_network import (ChessModel, fen_to_tensor, move_sequence_to_vector,
                             flip_square, MODEL_PATH, DEVICE)
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
    Scores are computed by multiplying the from-square and to-square
    probabilities, with board flipping applied for Black.
    """
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

    scored = []
    for move in board.legal_moves:
        from_sq = flip_square(move.from_square) if is_black else move.from_square
        to_sq   = flip_square(move.to_square)   if is_black else move.to_square
        score   = float(from_probs[from_sq]) * float(to_probs[to_sq])
        scored.append((score, move))

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
    scores = np.array([s for s, _ in scored])
    moves  = [m for _, m in scored]

    if temperature == 0.0:
        return moves[0].uci()

    scores = np.power(scores, 1.0 / temperature)
    total  = scores.sum()
    if total == 0:
        return np.random.choice(moves).uci()

    probs  = scores / total
    chosen = np.random.choice(len(moves), p=probs)
    return moves[chosen].uci()


def predict_next_move_with_search(model: ChessModel, board: chess.Board,
                                  top_n: int = 5,
                                  depth: int = 2) -> str | None:
    """
    Hybrid prediction: neural network narrows candidates, alphabeta picks the best.

    How it works:
      1. The neural network scores all legal moves and picks the top_n most
         Magnus-like candidates. This keeps the style in the opening while
         reducing the search space dramatically.
      2. Alphabeta searches each candidate to the given depth, evaluating
         consequences using the heuristic evaluation function.
      3. The move with the best search score is returned.

    Args:
        model  : trained ChessModel
        board  : current board position
        top_n  : how many NN candidates to search (5 is a good balance —
                 more candidates = more tactical coverage but slower)
        depth  : alphabeta search depth (2 = fast, 3 = stronger but slower,
                 4 = slow, only use in endgame with few pieces)

    Returns the predicted move as a UCI string.

    Why this works better than pure NN in the middlegame:
      The NN learned what Magnus played, but has no concept of consequences.
      Alphabeta explicitly evaluates future positions, so it avoids blunders
      like hanging pieces — the most common failure mode of the pure NN.
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
    side       = board.turn  # True = White, False = Black

    for move in candidates:
        board.push(move)
        # Search from opponent's perspective (negamax style)
        eval_score = -_alphabeta_search(
            board, depth - 1,
            float('-inf'), float('inf')
        )
        board.pop()

        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move

    return best_move.uci()


def _evaluate_board(board: chess.Board) -> float:
    """
    Evaluate the board from the perspective of the side to move.
    Positive = good for the side to move, negative = bad.
    Wraps chess_player.evaluate_helper + heuristics.evaluate.
    """
    raw = heuristics.evaluate(chess_player.evaluate_helper(board))
    # evaluate() returns score from White's perspective —
    # flip sign if it's Black's turn so it's always "higher = better for me"
    return raw if board.turn == chess.WHITE else -raw



def _alphabeta_search(board: chess.Board, depth: int,
                      alpha: float, beta: float) -> float:
    """
    Standard alphabeta search returning score from the perspective of
    the side to move at this node. Used internally by
    predict_next_move_with_search.
    """
    if board.is_checkmate():
        return -10000 + depth  # mated — shorter mate scores higher

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if depth == 0:
        return _evaluate_board(board)

    best = float('-inf')
    for move in board.legal_moves:
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

    print("\n--- Pure NN prediction (opening) ---")
    move = predict_next_move(model, board)
    print(f"Predicted first move: {move}")

    print("\n--- NN + search prediction (depth=2, top 5 candidates) ---")
    move = predict_next_move_with_search(model, board, top_n=5, depth=2)
    print(f"Predicted first move: {move}")