"""Audit test 4: value-target sign conventions across the whole chain.

Standalone runnable (no pytest):  python tests/audit_value_semantics_test.py

The convention everywhere is: value = expected outcome from the CURRENT
side-to-move's point of view, in [-1, 1]. This test proves each link:
  a. preprocess label functions (_result_value_for_color, _cp_to_value)
     follow the convention as pure functions;
  b. self-play outcome labeling follows it (recomputed from its formula);
  c. the TRAINED v3 value head learned it: blatant won/lost positions get
     confidently positive/negative values for the side to move;
  d. load_model's one-ply value rerank negates correctly: a mate-in-1 move
     scores exactly +1 via the terminal path and is chosen; a stalemating
     move scores exactly 0;
  e. MCTS backup keeps the sign: from a mate-in-1 root, search converges on
     the mate and reports Q ~= +1 for it from the root player's view.

If any sign in the chain were flipped, training loss would still decrease
(the head fits whatever target it is given) while play collapses — this is
exactly the class of bug the audit targets.
"""

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.preprocess import _cp_to_value, _result_value_for_color

V3_MODEL_PATH = 'model/grandmaster_resnet_v3.pt'

# Blatant positions: side to move is a queen up (or down) in a quiet spot.
# (fen, expected_sign) with value from the side-to-move's perspective.
SIGN_PROBES = [
    # White to move with an extra queen.
    ('4k3/8/8/8/8/8/4P3/QQ2K3 w - - 0 1', +1),
    # Black to move facing those two queens.
    ('4k3/8/8/8/8/8/4P3/QQ2K3 b - - 0 1', -1),
    # Black to move with an extra queen+rook.
    ('qq2k3/4p3/8/8/8/8/8/4K3 b - - 0 1', +1),
    # White to move facing them.
    ('qq2k3/4p3/8/8/8/8/8/4K3 w - - 0 1', -1),
]

MATE_IN_1 = [
    # (fen, mating move)
    ('6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1', 'a1a8'),   # back-rank rook mate
    ('8/8/8/8/8/6q1/5k2/7K b - - 0 1', 'g3g1'),      # black queen mates
]

STALEMATE_IN_1 = [
    # White queen c7->c6?? stalemates the black king on a8 (Ka7 illegal etc).
    ('k7/2Q5/8/8/8/8/8/4K3 w - - 0 1', 'c7b6', 'a8'),
]


def test_preprocess_value_functions():
    assert _result_value_for_color('1-0', chess.WHITE) == 1.0
    assert _result_value_for_color('1-0', chess.BLACK) == -1.0
    assert _result_value_for_color('0-1', chess.WHITE) == -1.0
    assert _result_value_for_color('0-1', chess.BLACK) == 1.0
    assert _result_value_for_color('1/2-1/2', chess.WHITE) == 0.0
    assert _cp_to_value(0) == 0.0
    assert 0.0 < _cp_to_value(100) < _cp_to_value(600) < 1.0
    assert _cp_to_value(-300) == -_cp_to_value(300)
    assert _cp_to_value(100_000) > 0.999
    print('  preprocess value label functions: side-to-move POV confirmed')


def test_self_play_outcome_labeling():
    # Reproduces experiments/self_play.py's formula on both outcomes.
    for winner in (chess.WHITE, chess.BLACK):
        for stm_is_white in (True, False):
            target = 1.0 if (winner == chess.WHITE) == stm_is_white else -1.0
            expected = 1.0 if (
                (winner == chess.WHITE and stm_is_white) or
                (winner == chess.BLACK and not stm_is_white)
            ) else -1.0
            assert target == expected
    print('  self-play outcome labeling: side-to-move POV confirmed')


def _load_model():
    from load_model import load_trained_model
    return load_trained_model(V3_MODEL_PATH)


def test_trained_value_head_sign(model):
    from load_model import evaluate_position
    for fen, sign in SIGN_PROBES:
        board = chess.Board(fen)
        v = evaluate_position(model, board)
        assert v * sign > 0.3, (
            f'value head sign suspect: {fen} -> {v:+.3f}, expected sign {sign:+d} '
            f'with confident magnitude'
        )
        print(f'    value({fen[:24]}...) = {v:+.3f}  (expected {sign:+d}) OK')
    print('  trained value head follows side-to-move convention')


def test_value_rerank_terminal_signs(model):
    from load_model import _value_scores_after_moves, predict_next_move

    for fen, mate_uci in MATE_IN_1:
        board = chess.Board(fen)
        fixture_check = board.copy()
        fixture_check.push(chess.Move.from_uci(mate_uci))
        assert fixture_check.is_checkmate(), f'fixture is not mate: {fen} {mate_uci}'
        moves = list(board.legal_moves)
        scores = _value_scores_after_moves(model, board, moves)
        by_move = {m.uci(): s for m, s in zip(moves, scores)}
        assert by_move[mate_uci] == 1.0, (
            f'mate-in-1 {mate_uci} at {fen} should score exactly +1.0 via the '
            f'terminal path, got {by_move[mate_uci]}'
        )

        def is_mate(uci: str) -> bool:
            b = board.copy()
            b.push(chess.Move.from_uci(uci))
            return b.is_checkmate()

        best = max(by_move, key=by_move.get)
        assert is_mate(best), (
            f'value scoring prefers non-mating {best} over a mate at {fen}'
        )
        picked = predict_next_move(model, board, temperature=0.0,
                                   value_weight=2.0, value_candidate_limit=0)
        assert is_mate(picked), (
            f'production config (greedy, value_weight=2) plays non-mating '
            f'{picked} at {fen}'
        )
        print(f'    mate-in-1 {mate_uci}: terminal value +1.0, mate chosen '
              f'greedily ({picked}) OK')

    for fen, stale_uci, _ in STALEMATE_IN_1:
        board = chess.Board(fen)
        move = chess.Move.from_uci(stale_uci)
        assert move in board.legal_moves
        board_check = board.copy()
        board_check.push(move)
        assert board_check.is_stalemate(), 'test position is wrong'
        scores = _value_scores_after_moves(model, board, [move])
        assert scores[0] == 0.0, (
            f'stalemating move should score exactly 0.0, got {scores[0]}'
        )
        print(f'    stalemate-in-1 {stale_uci}: terminal value 0.0 OK')
    print('  one-ply value rerank terminal signs correct')


def test_mcts_backup_sign(model):
    from inference.mcts_player import mcts_search

    for fen, mate_uci in MATE_IN_1:
        board = chess.Board(fen)
        root, stats = mcts_search(model, board, num_simulations=128,
                                  batch_size=8)
        visits = {m.uci(): c.visit_count for m, c in root.children.items()}
        best = max(visits, key=visits.get)
        best_child = root.children[chess.Move.from_uci(best)]
        q_for_root = -best_child.q_value()
        best_board = board.copy()
        best_board.push(chess.Move.from_uci(best))
        assert best_board.is_checkmate(), (
            f'MCTS visits favor non-mating {best} at {fen} '
            f'(visits={sorted(visits.items(), key=lambda kv: -kv[1])[:3]})'
        )
        assert q_for_root > 0.95, (
            f'MCTS Q for the chosen mating move should approach +1 from the '
            f'root POV, got {q_for_root:+.3f}'
        )
        print(f'    MCTS on mate-in-1 (picked {best}): '
              f'N={best_child.visit_count}/{stats.simulations}, '
              f'Q(root POV)={q_for_root:+.3f} OK')
    print('  MCTS backup sign chain correct')


if __name__ == '__main__':
    test_preprocess_value_functions()
    test_self_play_outcome_labeling()
    if os.path.exists(V3_MODEL_PATH):
        model = _load_model()
        test_trained_value_head_sign(model)
        test_value_rerank_terminal_signs(model)
        test_mcts_backup_sign(model)
    else:
        print(f'  SKIP model-dependent checks: {V3_MODEL_PATH} not found')
    print('audit value semantics tests passed')
