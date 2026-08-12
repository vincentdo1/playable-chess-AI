import os
import random
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (
    BOARD_CHANNELS, BOARD_CHANNELS_V3, MOVE_VOCAB_SIZE_V3,
    V3_EXTRA_CHANNELS, board_to_tensor_v3, fen_to_tensor,
    legal_policy_indices_v3, move_to_policy_index_v3, policy_index_to_move_v3,
)


def test_roundtrip_random_games():
    """Every legal move in random playouts must encode uniquely and decode back."""
    rng = random.Random(42)
    for game in range(40):
        board = chess.Board()
        for _ in range(120):
            if board.is_game_over():
                break
            flip = board.turn == chess.BLACK
            indices = set()
            for move in board.legal_moves:
                idx = move_to_policy_index_v3(move, flip=flip)
                assert 0 <= idx < MOVE_VOCAB_SIZE_V3
                assert idx not in indices, f'collision at {board.fen()} {move}'
                indices.add(idx)
                decoded = policy_index_to_move_v3(idx, flip=flip)
                assert decoded == move, (
                    f'{board.fen()} {move.uci()} -> {idx} -> {decoded.uci()}'
                )
            board.push(rng.choice(list(board.legal_moves)))


def test_promotions_roundtrip():
    # White and black promotions incl. captures and underpromotions.
    for fen, ucis in [
        ('rnbq1bnr/ppppkPpp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQ - 1 5',
         ['f7g8q', 'f7g8n', 'f7g8r', 'f7g8b']),
        ('rnbqkbnr/pppp1ppp/8/8/8/8/PPPPKPpp/RNBQ1BNR b kq - 1 5',
         ['g2h1q', 'g2h1n', 'g2f1q', 'g2f1b']),
    ]:
        board = chess.Board(fen)
        flip = board.turn == chess.BLACK
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f'{uci} not legal in {fen}'
            idx = move_to_policy_index_v3(move, flip=flip)
            assert policy_index_to_move_v3(idx, flip=flip) == move


def test_castling_roundtrip():
    board = chess.Board(
        'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8'
    )
    for color_board in (board, board.mirror()):
        flip = color_board.turn == chess.BLACK
        for uci in ('e1g1', 'e1c1') if not flip else ('e8g8', 'e8c8'):
            move = chess.Move.from_uci(uci)
            assert move in color_board.legal_moves
            idx = move_to_policy_index_v3(move, flip=flip)
            assert policy_index_to_move_v3(idx, flip=flip) == move


def test_board_planes_v3():
    board = chess.Board()
    t = board_to_tensor_v3(board)
    assert t.shape == (8, 8, BOARD_CHANNELS_V3)
    assert np.array_equal(t[:, :, :BOARD_CHANNELS], fen_to_tensor(board.fen()))
    assert t[:, :, V3_EXTRA_CHANNELS['halfmove_clock']].max() == 0.0
    assert t[:, :, V3_EXTRA_CHANNELS['repetition_1']].max() == 0.0

    # Halfmove clock plane scales with the FEN clock.
    board2 = chess.Board(
        'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 50 30'
    )
    t2 = board_to_tensor_v3(board2)
    assert abs(t2[0, 0, V3_EXTRA_CHANNELS['halfmove_clock']] - 0.5) < 1e-6

    # Repetition planes fire after shuffling knights back and forth.
    board3 = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'):
        board3.push(chess.Move.from_uci(uci))
    t3 = board_to_tensor_v3(board3)
    assert t3[0, 0, V3_EXTRA_CHANNELS['repetition_1']] == 1.0
    assert t3[0, 0, V3_EXTRA_CHANNELS['repetition_2']] == 1.0


def test_legal_indices_v3_consistency():
    board = chess.Board(
        'r2q1rk1/pp2bppp/2n1bn2/2pp4/8/2NP1NP1/PPQ1PPBP/R1B2RK1 w - - 6 15'
    )
    indices = legal_policy_indices_v3(board, flip=False)
    assert len(indices) == board.legal_moves.count()
    assert len(set(indices.tolist())) == len(indices)


def test_model_forward_shapes():
    import torch
    from neural_network import ChessModelV3
    model = ChessModelV3()
    model.eval()
    board = torch.zeros((2, BOARD_CHANNELS_V3, 8, 8))
    with torch.no_grad():
        policy, value = model(board)
    assert policy.shape == (2, MOVE_VOCAB_SIZE_V3)
    assert value.shape == (2,)
    assert float(value.abs().max()) <= 1.0


if __name__ == '__main__':
    test_roundtrip_random_games()
    test_promotions_roundtrip()
    test_castling_roundtrip()
    test_board_planes_v3()
    test_legal_indices_v3_consistency()
    test_model_forward_shapes()
    print('v3 encoding tests passed')
