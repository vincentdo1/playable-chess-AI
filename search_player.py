"""Inference-time alpha-beta search backed by the trained ChessModel.

Two ideas combined:
  - Policy head supplies move ordering, so alpha-beta cuts off branches fast.
  - Value head supplies leaf evaluations, replacing hand-crafted heuristics.

Plus the standard alpha-beta machinery that buys real Elo:
  iterative deepening, transposition table, quiescence on captures,
  Zobrist-keyed caching of network outputs so repeated positions cost zero.

Public entry point: ``search_best_move(model, board, ...)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import chess.polyglot
import numpy as np
import torch

from neural_network import ChessModel, move_to_policy_index
from load_model import _position_arrays


MATE_SCORE = 30000
MATE_THRESHOLD = MATE_SCORE - 1000
VALUE_TO_CP = 600.0

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

_PIECE_VALUE_CP = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}


def _value_to_cp(value: float) -> float:
    """Invert the value head's tanh(cp/600) so leaf scores live on a cp scale."""
    v = max(-0.999, min(0.999, value))
    return math.atanh(v) * VALUE_TO_CP


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    nn_calls: int = 0
    best_score_cp: float = 0.0
    completed_depth: int = 0


class NNEvaluator:
    """Single point of contact with the network; caches by Zobrist hash.

    Returns (policy_logits over MOVE_VOCAB_SIZE classes, value in [-1, 1] from side-to-move).
    Positions repeat constantly in alpha-beta (move ordering revisits the root,
    transposition collisions, qsearch back-references), so caching matters.
    """

    def __init__(self, model: ChessModel, stats: SearchStats | None = None):
        self.model = model
        self.device = next(model.parameters()).device
        self._cache: dict[int, tuple[np.ndarray, float]] = {}
        self.stats = stats

    def evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        key = chess.polyglot.zobrist_hash(board)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        board_arr, move_arr = _position_arrays(board)
        board_t = (
            torch.from_numpy(np.asarray(board_arr))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        move_t = (
            torch.from_numpy(np.asarray(move_arr))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            policy_logits, value = self.model(board_t, move_t)
        result = (
            policy_logits[0].detach().cpu().numpy(),
            float(value[0].item()),
        )
        self._cache[key] = result
        if self.stats is not None:
            self.stats.nn_calls += 1
        return result


def _terminal_score_cp(board: chess.Board, ply: int) -> int | None:
    """Score from side-to-move's POV for terminal nodes, else None."""
    if board.is_checkmate():
        # The side to move is the one being mated.
        return -(MATE_SCORE - ply)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.can_claim_fifty_moves()
    ):
        return 0
    return None


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA-ish ordering signal for tactical moves."""
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    victim_val = _PIECE_VALUE_CP.get(victim.piece_type, 0) if victim else 100
    attacker_val = _PIECE_VALUE_CP.get(attacker.piece_type, 0) if attacker else 0
    return victim_val - attacker_val


def _ordered_moves(
    evaluator: NNEvaluator,
    board: chess.Board,
    hash_move: chess.Move | None,
) -> list[chess.Move]:
    """Sort legal moves: hash move first, then by policy logit + tactical bonuses."""
    legal = list(board.legal_moves)
    if not legal:
        return legal

    policy_logits, _ = evaluator.evaluate(board)
    is_black = (board.turn == chess.BLACK)

    scored = []
    for move in legal:
        score = float(policy_logits[move_to_policy_index(move, flip=is_black)])
        if board.is_capture(move):
            # Capture bonus is tuned to roughly match the policy logit scale
            # so a good capture beats a moderately-likely quiet move.
            score += 4.0 + 0.005 * _capture_score(board, move)
        if move.promotion == chess.QUEEN:
            score += 6.0
        elif move.promotion is not None:
            score += 2.0
        if board.gives_check(move):
            score += 1.5
        scored.append((score, move))

    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [m for _, m in scored]
    if hash_move is not None and hash_move in ordered:
        ordered.remove(hash_move)
        ordered.insert(0, hash_move)
    return ordered


def _quiescence(
    evaluator: NNEvaluator,
    board: chess.Board,
    alpha: float,
    beta: float,
    ply: int,
    stats: SearchStats,
) -> float:
    """Extend search through captures so the leaf eval lands on a quiet position."""
    stats.qnodes += 1

    terminal = _terminal_score_cp(board, ply)
    if terminal is not None:
        return terminal

    _, value = evaluator.evaluate(board)
    stand_pat = _value_to_cp(value)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    tactical_moves = [
        m for m in board.legal_moves
        if board.is_capture(m) or m.promotion is not None
    ]
    if not tactical_moves:
        return alpha

    tactical_moves.sort(
        key=lambda m: (
            _capture_score(board, m)
            + (900 if m.promotion == chess.QUEEN else 0)
        ),
        reverse=True,
    )

    for move in tactical_moves:
        board.push(move)
        score = -_quiescence(evaluator, board, -beta, -alpha, ply + 1, stats)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _negamax(
    evaluator: NNEvaluator,
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    ply: int,
    tt: dict,
    stats: SearchStats,
    deadline: float | None,
) -> float:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError()

    stats.nodes += 1

    terminal = _terminal_score_cp(board, ply)
    if terminal is not None:
        return terminal

    key = chess.polyglot.zobrist_hash(board)
    hash_move: chess.Move | None = None
    tt_entry = tt.get(key)
    if tt_entry is not None:
        tt_depth, tt_score, tt_flag, tt_move = tt_entry
        hash_move = tt_move
        if tt_depth >= depth:
            stats.tt_hits += 1
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

    if depth <= 0:
        return _quiescence(evaluator, board, alpha, beta, ply, stats)

    original_alpha = alpha
    best_score = -MATE_SCORE
    best_move: chess.Move | None = None

    for move in _ordered_moves(evaluator, board, hash_move):
        board.push(move)
        score = -_negamax(
            evaluator, board, depth - 1, -beta, -alpha, ply + 1, tt, stats, deadline,
        )
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    if best_score <= original_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    tt[key] = (depth, best_score, flag, best_move)
    return best_score


def search_best_move(
    model: ChessModel,
    board: chess.Board,
    max_depth: int = 4,
    time_limit: float | None = None,
    verbose: bool = False,
) -> chess.Move | None:
    """Iterative-deepening alpha-beta with the network as evaluator and orderer.

    Stops at ``max_depth`` plies, or earlier when ``time_limit`` seconds elapse.
    The deepest fully-completed iteration's best move is returned; a partial
    iteration aborted by the timer is discarded.
    """
    if board.is_game_over(claim_draw=True):
        return None
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    stats = SearchStats()
    evaluator = NNEvaluator(model, stats=stats)
    tt: dict = {}
    deadline = time.monotonic() + time_limit if time_limit else None

    best_move = legal[0]
    best_score = -MATE_SCORE

    for depth in range(1, max_depth + 1):
        try:
            root_key = chess.polyglot.zobrist_hash(board)
            root_tt = tt.get(root_key)
            hash_move = root_tt[3] if root_tt is not None else None

            alpha, beta = -MATE_SCORE, MATE_SCORE
            iter_best_move: chess.Move | None = None
            iter_best_score = -MATE_SCORE

            for move in _ordered_moves(evaluator, board, hash_move):
                board.push(move)
                score = -_negamax(
                    evaluator, board, depth - 1, -beta, -alpha, 1, tt, stats, deadline,
                )
                board.pop()
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move = move
                if iter_best_score > alpha:
                    alpha = iter_best_score

            if iter_best_move is not None:
                best_move = iter_best_move
                best_score = iter_best_score
                stats.completed_depth = depth
                stats.best_score_cp = best_score
                if verbose:
                    print(
                        f"  depth {depth}: best={best_move.uci()} "
                        f"score={best_score:+.0f}cp nodes={stats.nodes} "
                        f"qnodes={stats.qnodes} nn={stats.nn_calls} "
                        f"tt_hits={stats.tt_hits}"
                    )

            if abs(best_score) > MATE_THRESHOLD:
                break
        except TimeoutError:
            if verbose:
                print(
                    f"  depth {depth}: timed out, using depth {stats.completed_depth}"
                )
            break

    return best_move


def search_predict_uci(
    model: ChessModel,
    board: chess.Board,
    max_depth: int = 4,
    time_limit: float | None = None,
    verbose: bool = False,
) -> str | None:
    """Convenience wrapper that returns the chosen move as a UCI string."""
    move = search_best_move(
        model, board, max_depth=max_depth, time_limit=time_limit, verbose=verbose,
    )
    return move.uci() if move is not None else None


if __name__ == '__main__':
    from load_model import load_trained_model

    print("Loading model...")
    model = load_trained_model()
    board = chess.Board()

    print("\n--- Search depth 3 from startpos ---")
    move = search_best_move(model, board, max_depth=3, verbose=True)
    print(f"Picked: {move.uci() if move else None}")

    print("\n--- Search depth 4 from a tactical middlegame ---")
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    move = search_best_move(model, board, max_depth=4, verbose=True)
    print(f"Picked: {move.uci() if move else None}")
