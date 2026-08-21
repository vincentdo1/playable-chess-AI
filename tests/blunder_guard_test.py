import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.blunder_guard import filter_scored_moves


def test_vetoes_hanging_queen():
    # The e6 pawn can recapture Qxd5.
    board = chess.Board(
        'rnbqkbnr/ppp2ppp/4p3/3p4/8/3P1Q2/PPP1PPPP/RNB1KBNR w KQkq - 0 9'
    )
    qxd5 = chess.Move.from_uci('f3d5')  # captures d5 pawn, e6 pawn recaptures
    quiet = chess.Move.from_uci('e2e4')
    assert qxd5 in board.legal_moves and quiet in board.legal_moves

    scored = [(0.0, qxd5), (-1.0, quiet)]
    safe = filter_scored_moves(board, scored, depth=2, margin_cp=150)
    safe_moves = [m for _, m in safe]
    assert qxd5 not in safe_moves, 'guard should veto capturing a defended pawn with the queen'
    assert quiet in safe_moves


def test_keeps_reasonable_moves():
    board = chess.Board(
        'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 10'
    )
    moves = list(board.legal_moves)[:6]
    scored = [(-float(i), m) for i, m in enumerate(moves)]
    safe = filter_scored_moves(board, scored, depth=2, margin_cp=150)
    assert len(safe) >= 1
    scores = [s for s, _ in safe]
    assert scores == sorted(scores, reverse=True)


def test_opening_passthrough():
    board = chess.Board()  # fullmove 1 < min_fullmove
    moves = list(board.legal_moves)
    scored = [(-float(i), m) for i, m in enumerate(moves)]
    assert filter_scored_moves(board, scored) == scored


if __name__ == '__main__':
    test_vetoes_hanging_queen()
    test_keeps_reasonable_moves()
    test_opening_passthrough()
    print('blunder_guard tests passed')
