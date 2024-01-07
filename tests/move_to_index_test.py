import unittest
import numpy as np
import sys, os
import chess
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import move_to_index

class TestMoveToIndex(unittest.TestCase):
    def test_standard_move(self):
        board = chess.Board()
        move = chess.Move.from_uci('e2e4')
        move_index = move_to_index(move, board)
        self.assertEqual(len(move_index), 132)
        self.assertEqual(move_index[chess.E2], 1)
        self.assertEqual(move_index[64 + chess.E4], 1)
        self.assertEqual(np.sum(move_index), 2)

    def test_pawn_promotion_queen(self):
        board = chess.Board('4k3/7P/8/8/8/8/8/4K3 w - - 0 1')  # Position set up for promotion
        move = chess.Move.from_uci('h7h8q')  # Pawn promotes to queen
        move_index = move_to_index(move, board)
        self.assertEqual(move_index[chess.H7], 1)
        self.assertEqual(move_index[64 + chess.H8], 1)
        self.assertEqual(move_index[131], 1)  # Index for queen promotion
        self.assertEqual(np.sum(move_index), 3)

    def test_pawn_promotion_knight(self):
        board = chess.Board('8/P7/8/8/8/8/8/4K3 w - - 0 1')  # Position set up for promotion
        move = chess.Move.from_uci('a7a8n')  # Pawn promotes to knight
        move_index = move_to_index(move, board)
        self.assertEqual(move_index[chess.A7], 1)
        self.assertEqual(move_index[64 + chess.A8], 1)
        self.assertEqual(move_index[128], 1)  # Index for knight promotion
        self.assertEqual(np.sum(move_index), 3)

    def test_invalid_move(self):
        board = chess.Board()
        move = chess.Move.from_uci('e1e5')  # Assuming an invalid move
        with self.assertRaises(ValueError):
            move_to_index(move, board)

if __name__ == '__main__':
    unittest.main()
