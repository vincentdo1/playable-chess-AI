import unittest
import numpy as np
import sys, os
import chess
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import move_to_vector

class TestMoveToVector(unittest.TestCase):

    def test_valid_move(self):
        move = chess.Move(from_square = chess.E2, to_square = chess.E4)  # Replace ChessMove with appropriate move representation
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 2)
        self.assertEqual(vector[chess.E2], 1)
        self.assertEqual(vector[64 + chess.E4], 1)

    def test_move_with_capture(self):
        move = chess.Move(from_square=chess.E5, to_square=chess.D6)
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 2)
        self.assertEqual(vector[chess.E5], 1)
        self.assertEqual(vector[64 + chess.D6], 1)

    def test_castling_move(self):
        move = chess.Move(from_square=chess.E1, to_square=chess.G1)  # Kingside castling for white
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 2)
        self.assertEqual(vector[chess.E1], 1)
        self.assertEqual(vector[64 + chess.G1], 1)

    def test_castling_move(self):
        move = chess.Move(from_square=chess.E1, to_square=chess.G1)  # Kingside castling for white
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 2)
        self.assertEqual(vector[chess.E1], 1)
        self.assertEqual(vector[64 + chess.G1], 1)

    def test_pawn_promotion_move(self):
        move = chess.Move(from_square=chess.A7, to_square=chess.A8, promotion=chess.QUEEN)  # Pawn promotion to queen
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 3)
        self.assertEqual(vector[chess.A7], 1)
        self.assertEqual(vector[64 + chess.A8], 1)
        self.assertEqual(vector[131], 1)

    def test_edge_move(self):
        move = chess.Move(from_square=chess.H8, to_square=chess.F8)
        vector = move_to_vector(move)
        self.assertEqual(len(vector), 132)
        self.assertEqual(np.sum(vector), 2)
        self.assertEqual(vector[chess.H8], 1)
        self.assertEqual(vector[64 + chess.F8], 1)

if __name__ == '__main__':
    unittest.main()
