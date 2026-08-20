"""Tactical safety net for the raw policy player.

The supervised policy blunders (>=300cp) on ~14% of moves after move 20 (see
evaluation/diagnose_phase_degradation.py). This module vetoes the worst of
those without changing the model: the network's top candidates are scored with
the existing shallow alpha-beta (chess_player/heuristics), and a candidate is
played only if it doesn't lose material relative to the best candidate by more
than a margin. The model still chooses among tactically-sound moves, so style
is preserved.

Cost: GUARD_CANDIDATES shallow searches per move (pure Python). Depth 2 is
~tens of ms; depth 3 catches more tactics but costs a few hundred ms.
"""

from __future__ import annotations

import chess

import chess_player

# A candidate is vetoed when shallow search says it scores worse than the
# best candidate by more than this many centipawns. ~a minor piece's worth
# of slack keeps positional model choices alive while stopping hung pieces.
DEFAULT_MARGIN_CP = 150
DEFAULT_DEPTH = 2
DEFAULT_CANDIDATES = 8
# The policy is near-GM in the opening (memorized theory, 0.7% blunder rate)
# and the crude heuristic eval only second-guesses good opening moves, so the
# guard stays off until the middlegame.
DEFAULT_MIN_FULLMOVE = 9


def _search_cp_after_move(board: chess.Board, move: chess.Move,
                          depth: int) -> float:
    """Shallow alpha-beta score after `move`, in cp from the mover's POV.

    Same convention as training/preprocess._search_score_after_move_cp:
    heuristics.evaluate is white-POV in pawns, so flip for black and x100.
    """
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
    """Filter (model_score, move) pairs down to tactically-safe ones.

    scored_moves must be sorted by model score, best first (the format
    returned by load_model._get_move_scores). Returns a non-empty subset in
    the same order: candidates within margin_cp of the best shallow-search
    score. Moves beyond max_candidates are dropped when the guard runs —
    they were already low-confidence, and searching them would be wasted
    time.
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
    # Defensive: margin filtering can't actually empty the list (the best
    # candidate always passes), but never return nothing.
    best_idx = search_scores.index(best_search)
    return [candidates[best_idx]]
