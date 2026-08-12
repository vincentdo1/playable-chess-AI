"""Audit tests 1+2: board encoding correctness and side-to-move orientation.

Standalone runnable (no pytest):  python tests/audit_encoding_orientation_test.py

What is proven here:
  1. fen_to_tensor / board_to_tensor_v3 are lossless: an independent decoder
     reconstructs the exact position (pieces, castling, legal en passant)
     from the tensor, for both perspectives, across random playouts.
  2. Side-to-move orientation is exactly self-consistent: a position P and
     its 180-degree-rotated color-swapped twin Q produce BYTE-IDENTICAL
     tensors and IDENTICAL policy label indices (v2 and v3 codecs). Any
     white/black asymmetry bug in the perspective frame breaks this.
  3. The v3 convolutional policy head's spatial frame matches the board
     planes: the from-square of every legal move's policy index lands on
     the tensor cell that holds the moving piece.
  4. End-to-end on the real trained v3 checkpoint: the inference path
     (load_model._get_move_scores) gives the same scores for P and Q with
     moves mapped through the rotation, and the same value.
"""

import os
import random
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (
    BOARD_CHANNELS, NUM_MOVE_PLANES, V3_EXTRA_CHANNELS,
    board_to_tensor_v3, fen_to_tensor, move_to_policy_index,
    move_to_policy_index_v3, piece_to_index,
)

V3_MODEL_PATH = 'model/grandmaster_resnet_v3.pt'

_NAME_TO_PIECE_TYPE = {
    'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
    'rook': chess.ROOK, 'queen': chess.QUEEN, 'king': chess.KING,
}


def decode_tensor(tensor, turn):
    """Independent inverse of fen_to_tensor: tensor + side-to-move -> board.

    Deliberately re-derives the coordinate convention from scratch so a bug
    in fen_to_tensor cannot hide behind a shared implementation.
    """
    flip = (turn == chess.BLACK)
    own_color = chess.BLACK if flip else chess.WHITE
    board = chess.Board(None)
    board.turn = turn
    for name, channel in piece_to_index.items():
        if 'castle' in name or name == 'ep':
            continue
        side, piece_name = name.split('_', 1)
        color = own_color if side == 'own' else (not own_color)
        rows, cols = np.nonzero(tensor[:, :, channel])
        for row, col in zip(rows, cols):
            rank = 7 - row if flip else row
            file = 7 - col if flip else col
            board.set_piece_at(
                chess.square(file, rank),
                chess.Piece(_NAME_TO_PIECE_TYPE[piece_name], color),
            )

    castling = ''
    own_k, own_q = ('k', 'q') if own_color == chess.BLACK else ('K', 'Q')
    opp_k, opp_q = ('K', 'Q') if own_color == chess.BLACK else ('k', 'q')
    if tensor[0, 0, piece_to_index['own_kingside_castle']]:
        castling += own_k
    if tensor[0, 0, piece_to_index['own_queenside_castle']]:
        castling += own_q
    if tensor[0, 0, piece_to_index['opp_kingside_castle']]:
        castling += opp_k
    if tensor[0, 0, piece_to_index['opp_queenside_castle']]:
        castling += opp_q
    board.set_castling_fen(''.join(sorted(castling, key='KQkq'.index)) or '-')

    ep_rows, ep_cols = np.nonzero(tensor[:, :, piece_to_index['ep']])
    assert len(ep_rows) <= 1, 'more than one en-passant square in tensor'
    if len(ep_rows) == 1:
        rank = 7 - ep_rows[0] if flip else ep_rows[0]
        file = 7 - ep_cols[0] if flip else ep_cols[0]
        board.ep_square = chess.square(file, rank)
    return board


def random_positions(num_games=15, max_plies=100, seed=1234):
    rng = random.Random(seed)
    for _ in range(num_games):
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            yield board
            board.push(rng.choice(list(board.legal_moves)))


def test_encoding_lossless_roundtrip():
    positions = castled = eps = 0
    for board in random_positions():
        fen = board.fen()
        tensor = fen_to_tensor(fen)
        decoded = decode_tensor(tensor, board.turn)

        ref = chess.Board(fen)
        assert decoded.piece_map() == ref.piece_map(), (
            f'piece mismatch at {fen}'
        )
        assert decoded.castling_rights == ref.castling_rights, (
            f'castling mismatch at {fen}: '
            f'{decoded.castling_xfen()} vs {ref.castling_xfen()}'
        )
        # fen_to_tensor sees board.fen(), which only writes the ep square
        # when an en-passant capture is actually legal.
        expected_ep = ref.ep_square if ref.has_legal_en_passant() else None
        assert decoded.ep_square == expected_ep, (
            f'ep mismatch at {fen}: {decoded.ep_square} vs {expected_ep}'
        )
        positions += 1
        if ref.castling_rights:
            castled += 1
        if expected_ep is not None:
            eps += 1
    assert positions > 500, f'too few positions exercised: {positions}'
    assert eps > 0, 'no en-passant positions were exercised'
    print(f'  encoding round-trip OK on {positions} positions '
          f'({castled} with castling rights, {eps} with legal ep)')


def rot180_color_swap(board: chess.Board) -> chess.Board:
    """Exact 180-degree rotation with colors and side-to-move swapped."""
    return board.mirror().transform(chess.flip_horizontal)


def test_orientation_invariance_encoding():
    """encode(P, stm view) must equal encode(rot180cs(P), stm view) exactly."""
    checked = 0
    for board in random_positions(num_games=12, seed=99):
        if board.castling_rights:
            # Castling under horizontal flip is a Chess960-style corner case
            # of python-chess transforms, not of our encoder; skip.
            continue
        p = chess.Board(board.fen())      # stackless: repetition planes zero
        q = rot180_color_swap(p)
        if q.status() != chess.STATUS_VALID:
            continue

        tp = board_to_tensor_v3(p)
        tq = board_to_tensor_v3(q)
        assert np.array_equal(tp, tq), (
            f'v3 tensors differ for rot180+color-swap twins:\n'
            f'  P={p.fen()}\n  Q={q.fen()}'
        )

        flip_p = p.turn == chess.BLACK
        flip_q = q.turn == chess.BLACK
        p_indices_v3 = {}
        p_indices_v2 = {}
        for m in p.legal_moves:
            p_indices_v3[m] = move_to_policy_index_v3(m, flip=flip_p)
            p_indices_v2[m] = move_to_policy_index(m, flip=flip_p)
        q_moves = set(q.legal_moves)
        assert len(q_moves) == len(p_indices_v3), (
            f'legal move count differs: {p.fen()} vs {q.fen()}'
        )
        for m, idx_v3 in p_indices_v3.items():
            m_rot = chess.Move(
                63 - m.from_square, 63 - m.to_square, promotion=m.promotion
            )
            assert m_rot in q_moves, (
                f'rotated move {m_rot.uci()} not legal in twin of {p.fen()}'
            )
            assert move_to_policy_index_v3(m_rot, flip=flip_q) == idx_v3, (
                f'v3 index differs for {m.uci()} / {m_rot.uci()} at {p.fen()}'
            )
            assert move_to_policy_index(m_rot, flip=flip_q) == p_indices_v2[m], (
                f'v2 index differs for {m.uci()} / {m_rot.uci()} at {p.fen()}'
            )
        checked += 1
    assert checked > 200, f'too few twin positions exercised: {checked}'
    print(f'  orientation invariance OK on {checked} rot180+color-swap pairs '
          f'(tensors byte-identical, v2+v3 label indices identical)')


def test_policy_head_spatial_alignment():
    """Policy index from-square must address the tensor cell of the mover."""
    checked_moves = 0
    for board in random_positions(num_games=6, seed=7):
        flip = board.turn == chess.BLACK
        tensor = board_to_tensor_v3(board)
        for move in board.legal_moves:
            idx = move_to_policy_index_v3(move, flip=flip)
            from_sq = idx // NUM_MOVE_PLANES
            row, col = divmod(from_sq, 8)
            piece = board.piece_at(move.from_square)
            channel = piece_to_index[f'own_{_piece_name(piece)}']
            assert tensor[row, col, channel] == 1.0, (
                f'{board.fen()} {move.uci()}: policy from-square ({row},{col}) '
                f'does not hold the moving {piece} in the board tensor'
            )
            checked_moves += 1
    assert checked_moves > 3000, f'too few moves exercised: {checked_moves}'
    print(f'  conv policy head spatial alignment OK on {checked_moves} moves')


def _piece_name(piece: chess.Piece) -> str:
    return chess.piece_name(piece.piece_type)


def test_model_orientation_end_to_end():
    """Real v3 checkpoint: P and its rot180cs twin get identical scores."""
    if not os.path.exists(V3_MODEL_PATH):
        print(f'  SKIP: {V3_MODEL_PATH} not found')
        return
    from load_model import load_trained_model, _get_move_scores, evaluate_position

    model = load_trained_model(V3_MODEL_PATH)
    pairs_checked = 0
    for board in random_positions(num_games=4, max_plies=60, seed=2024):
        if board.castling_rights or board.fullmove_number < 6:
            continue
        p = chess.Board(board.fen())
        q = rot180_color_swap(p)
        if q.status() != chess.STATUS_VALID or q.is_game_over():
            continue

        vp = evaluate_position(model, p)
        vq = evaluate_position(model, q)
        assert abs(vp - vq) < 1e-4, (
            f'value differs for twins: {vp} vs {vq} at {p.fen()}'
        )

        scores_p = {m: s for s, m in
                    _get_move_scores(model, p, value_weight=0.0)}
        scores_q = {m: s for s, m in
                    _get_move_scores(model, q, value_weight=0.0)}
        for m, s in scores_p.items():
            m_rot = chess.Move(
                63 - m.from_square, 63 - m.to_square, promotion=m.promotion
            )
            assert abs(scores_q[m_rot] - s) < 1e-3, (
                f'policy score differs for {m.uci()}/{m_rot.uci()} at '
                f'{p.fen()}: {s} vs {scores_q[m_rot]}'
            )
        pairs_checked += 1
        if pairs_checked >= 25:
            break
    assert pairs_checked >= 10, f'too few model pairs checked: {pairs_checked}'
    print(f'  end-to-end model orientation OK on {pairs_checked} twin pairs '
          f'(white-to-move and black-to-move inference paths agree exactly)')


if __name__ == '__main__':
    test_encoding_lossless_roundtrip()
    test_orientation_invariance_encoding()
    test_policy_head_spatial_alignment()
    test_model_orientation_end_to_end()
    print('audit encoding/orientation tests passed')
