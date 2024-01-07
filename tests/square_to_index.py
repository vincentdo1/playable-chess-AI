import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import square_to_index

class TestSquareToIndex(unittest.TestCase):

    def test_valid_inputs(self):
        # Test with valid inputs
        self.assertEqual(square_to_index('a1'), 56)
        self.assertEqual(square_to_index('h1'), 63)
        self.assertEqual(square_to_index('a8'), 0)
        self.assertEqual(square_to_index('h8'), 7)
        self.assertEqual(square_to_index('e4'), 36)

    def test_invalid_ranks(self):
        # Test with invalid ranks
        with self.assertRaises(ValueError):
            square_to_index('a9')
        with self.assertRaises(ValueError):
            square_to_index('b0')

    def test_invalid_files(self):
        # Test with invalid files
        with self.assertRaises(ValueError):
            square_to_index('i1')
        with self.assertRaises(ValueError):
            square_to_index('z3')

    def test_incorrect_format(self):
        # Test with incorrectly formatted inputs
        with self.assertRaises(ValueError):
            square_to_index('1a')
        with self.assertRaises(ValueError):
            square_to_index('8h')

    def test_special_characters(self):
        # Test with special characters
        with self.assertRaises(ValueError):
            square_to_index('*1')
        with self.assertRaises(ValueError):
            square_to_index('@2')

    def test_uppercase_inputs(self):
        # Test with uppercase inputs
        self.assertEqual(square_to_index('A1'), 56)
        self.assertEqual(square_to_index('H8'), 7)

    def test_edge_cases(self):
        # Test edge cases
        self.assertEqual(square_to_index('a8'), 0)
        self.assertEqual(square_to_index('h1'), 63)

if __name__ == '__main__':
    unittest.main()
