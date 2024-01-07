import unittest
import numpy as np
import sys, os
import chess
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import move_sequence_to_vector, square_to_index

class TestMoveSequenceToVector(unittest.TestCase):
    def test_standard_move_sequence(self):
        move_sequence = [chess.Move.from_uci('e2e4'), chess.Move.from_uci('e7e5'), chess.Move.from_uci('g1f3')]
        sequence_vector = move_sequence_to_vector(move_sequence)
        self.assertEqual(sequence_vector.shape, (10, 132))
        self.assertEqual(np.sum(sequence_vector[0]), 2)
        self.assertEqual(np.sum(sequence_vector[1]), 2)
        self.assertEqual(np.sum(sequence_vector[2]), 2)

    def test_pawn_promotion_sequence(self):
        move_sequence = [chess.Move.from_uci('e7e8q')]  # Pawn promotion to queen
        sequence_vector = move_sequence_to_vector(move_sequence)
        self.assertEqual(sequence_vector.shape, (10, 132))
        self.assertEqual(sequence_vector[0][square_to_index('e7')], 1)
        self.assertEqual(sequence_vector[0][64 + square_to_index('e8')], 1)
        self.assertEqual(sequence_vector[0][131], 1)  # Index for queen promotion

    def test_mixed_move_sequence(self):
        move_sequence = [chess.Move.from_uci('e2e4'), chess.Move.from_uci('e7e8q')]
        sequence_vector = move_sequence_to_vector(move_sequence)
        self.assertEqual(sequence_vector.shape, (10, 132))
        self.assertEqual(np.sum(sequence_vector[0]), 2)
        self.assertEqual(sequence_vector[1][square_to_index('e7')], 1)
        self.assertEqual(sequence_vector[1][64 + square_to_index('e8')], 1)
        self.assertEqual(sequence_vector[1][131], 1)  # Index for queen promotion

    def test_short_move_sequence(self):
        move_sequence = [chess.Move.from_uci('e2e4')]
        sequence_vector = move_sequence_to_vector(move_sequence)
        self.assertEqual(sequence_vector.shape, (10, 132))
        self.assertEqual(np.sum(sequence_vector[0]), 2)
        self.assertTrue(np.all(sequence_vector[1:] == 0))  # Remaining vectors should be zeros

    def test_long_move_sequence(self):
        move_sequence = [chess.Move.from_uci(uci) for uci in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'e1g1', 'f8c5', 'c2c3', 'd7d6', 'd2d4']]
        sequence_vector = move_sequence_to_vector(move_sequence)
        self.assertEqual(sequence_vector.shape, (10, 132))
        self.assertNotEqual(np.sum(sequence_vector[0]), 0)
        self.assertNotEqual(np.sum(sequence_vector[-1]), 0)
        for move_vector in sequence_vector[1:9]:
            self.assertEqual(np.sum(move_vector), 2, "Each move vector should have exactly two non-zero elements")

if __name__ == '__main__':
    unittest.main()


