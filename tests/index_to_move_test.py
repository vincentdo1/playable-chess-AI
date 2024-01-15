import unittest
import numpy as np
import sys, os
import chess
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from load_model import index_to_move

class TestIndexToMove(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()

    def test_normal_move(self):
        index = 20  # Assuming this index corresponds to a legal move like 'e2e4'
        move = index_to_move(index, self.board)
        self.assertEqual(move, 'e2e4')

    def test_pawn_promotion_to_queen(self):
        self.board.set_board_fen('8/P7/8/8/8/8/8/8 w - - 0 1')  # Set up a board just before pawn promotion
        index = 128  # Assuming this index corresponds to a7a8 promotion to queen
        move = index_to_move(index, self.board)
        self.assertEqual(move, 'a7a8q')

    def test_illegal_move(self):
        index = 62  # Assuming this index corresponds to an illegal move
        move = index_to_move(index, self.board)
        self.assertIsNone(move)

    def test_edge_case(self):
        # Assuming the index represents an edge case like 'a1h8' which is normally impossible
        index = 15
        move = index_to_move(index, self.board)
        self.assertIsNone(move)  # Expected to be None as it's an illegal move
if __name__ == '__main__':
    unittest.main()