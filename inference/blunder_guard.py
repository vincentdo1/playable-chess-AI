"""Filter policy candidates with a shallow tactical search."""

from __future__ import annotations

import chess

import chess_player

DEFAULT_MARGIN_CP = 150
DEFAULT_DEPTH = 2
DEFAULT_CANDIDATES = 8
# Skip openings, where the heuristic guard is least reliable.
DEFAULT_MIN_FULLMOVE = 9


def _search_cp_after_move(board: chess.Board, move: chess.Move,
                          depth: int) -> float:
    """Return a shallow-search score in centipawns from the mover's view."""
    mover = board.turn
    board.push(move)
    try:
        depth_remaining = max(0, depth - 1)
        white_score = chess_player.alphabetahelper(
            board.turn, board, depth_remaining, -10000, 10000
        )
    finally:
        board.pop()
    mover_score = white_score if mover == chess.WHITE else -white_score
    return float(mover_score) * 100.0


def filter_scored_moves(board: chess.Board, scored_moves,
                        depth: int = DEFAULT_DEPTH,
                        margin_cp: float = DEFAULT_MARGIN_CP,
                        max_candidates: int = DEFAULT_CANDIDATES,
                        min_fullmove: int = DEFAULT_MIN_FULLMOVE):
    """Keep moves within ``margin_cp`` of the best searched candidate.

    Input and output are ordered by model score. At most ``max_candidates``
    moves are searched.
    """
    if len(scored_moves) <= 1:
        return scored_moves
    if board.fullmove_number < min_fullmove:
        return scored_moves

    candidates = scored_moves[:max_candidates]
    search_scores = [
        _search_cp_after_move(board, move, depth)
        for _, move in candidates
    ]
    best_search = max(search_scores)

    safe = [
        pair for pair, search_cp in zip(candidates, search_scores)
        if search_cp >= best_search - margin_cp
    ]
    if safe:
        return safe
    best_idx = search_scores.index(best_search)
    return [candidates[best_idx]]
