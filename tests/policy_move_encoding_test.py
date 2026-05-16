import os
import sys
import unittest

import chess

sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import (
    legal_policy_indices,
    move_to_policy_index,
    policy_index_to_move,
)


class TestPolicyMoveEncoding(unittest.TestCase):
    def test_standard_move_round_trip(self):
        move = chess.Move.from_uci('e2e4')
        index = move_to_policy_index(move)
        self.assertEqual(policy_index_to_move(index), move)

    def test_promotion_round_trip(self):
        move = chess.Move.from_uci('a7a8q')
        index = move_to_policy_index(move)
        self.assertEqual(policy_index_to_move(index), move)

    def test_black_flip_round_trip(self):
        move = chess.Move.from_uci('e7e5')
        index = move_to_policy_index(move, flip=True)
        self.assertEqual(policy_index_to_move(index, flip=True), move)

    def test_legal_indices_include_en_passant(self):
        board = chess.Board()
        for uci in ['e2e4', 'a7a6', 'e4e5', 'd7d5']:
            board.push(chess.Move.from_uci(uci))

        ep_move = chess.Move.from_uci('e5d6')
        self.assertIn(ep_move, board.legal_moves)
        indices = legal_policy_indices(board)
        self.assertIn(move_to_policy_index(ep_move), indices)


if __name__ == '__main__':
    unittest.main()
