import os
import sys
import unittest

import chess

sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import move_to_policy_index, policy_index_to_move


class TestPolicyIndexToMove(unittest.TestCase):
    def test_normal_move(self):
        move = chess.Move.from_uci('e2e4')
        index = move_to_policy_index(move)
        self.assertEqual(policy_index_to_move(index), move)

    def test_pawn_promotion_to_queen(self):
        move = chess.Move.from_uci('a7a8q')
        index = move_to_policy_index(move)
        self.assertEqual(policy_index_to_move(index), move)

    def test_flipped_black_move(self):
        move = chess.Move.from_uci('e7e5')
        index = move_to_policy_index(move, flip=True)
        self.assertEqual(policy_index_to_move(index, flip=True), move)

    def test_out_of_range_index(self):
        with self.assertRaises(ValueError):
            policy_index_to_move(-1)


if __name__ == '__main__':
    unittest.main()
