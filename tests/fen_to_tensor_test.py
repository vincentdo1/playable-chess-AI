import unittest

import numpy as np

from neural_network import BOARD_CHANNELS, fen_to_tensor, piece_to_index


class TestFenToTensor(unittest.TestCase):
    def test_starting_position_white_perspective(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tensor = fen_to_tensor(fen, flip=False)

        self.assertEqual(tensor.shape, (8, 8, BOARD_CHANNELS))
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['own_pawn']]), 8)
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['opp_pawn']]), 8)
        self.assertEqual(tensor[0, 4, piece_to_index['own_king']], 1)
        self.assertEqual(tensor[7, 4, piece_to_index['opp_king']], 1)
        self.assertEqual(tensor[0, 1, piece_to_index['own_knight']], 1)
        self.assertEqual(tensor[7, 1, piece_to_index['opp_knight']], 1)
        self.assertEqual(
            np.sum(tensor[:, :, piece_to_index['own_kingside_castle']]), 64
        )
        self.assertEqual(
            np.sum(tensor[:, :, piece_to_index['opp_queenside_castle']]), 64
        )

    def test_starting_position_black_perspective(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        tensor = fen_to_tensor(fen)

        self.assertEqual(tensor.shape, (8, 8, BOARD_CHANNELS))
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['own_pawn']]), 8)
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['opp_pawn']]), 8)
        self.assertEqual(tensor[0, 3, piece_to_index['own_king']], 1)
        self.assertEqual(tensor[7, 3, piece_to_index['opp_king']], 1)
        self.assertEqual(tensor[0, 1, piece_to_index['own_knight']], 1)
        self.assertEqual(tensor[7, 1, piece_to_index['opp_knight']], 1)
        self.assertEqual(
            np.sum(tensor[:, :, piece_to_index['own_kingside_castle']]), 64
        )
        self.assertEqual(
            np.sum(tensor[:, :, piece_to_index['opp_queenside_castle']]), 64
        )

    def test_piece_channels_are_perspective_relative(self):
        white_to_move = fen_to_tensor("8/8/8/3q4/8/8/8/4K3 w - - 0 1")
        black_to_move = fen_to_tensor("8/8/8/3q4/8/8/8/4K3 b - - 0 1",
                                      flip=True)

        self.assertEqual(white_to_move[4, 3, piece_to_index['opp_queen']], 1)
        self.assertEqual(np.sum(white_to_move[:, :, piece_to_index['own_queen']]), 0)
        self.assertEqual(black_to_move[3, 4, piece_to_index['own_queen']], 1)
        self.assertEqual(np.sum(black_to_move[:, :, piece_to_index['opp_queen']]), 0)

    def test_en_passant_target(self):
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        tensor = fen_to_tensor(fen)

        self.assertEqual(tensor.shape, (8, 8, BOARD_CHANNELS))
        self.assertEqual(tensor[5, 3, piece_to_index['ep']], 1)
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['ep']]), 1)

    def test_flipped_en_passant_target(self):
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq d6 0 3"
        tensor = fen_to_tensor(fen, flip=True)

        self.assertEqual(tensor.shape, (8, 8, BOARD_CHANNELS))
        self.assertEqual(tensor[2, 4, piece_to_index['ep']], 1)
        self.assertEqual(np.sum(tensor[:, :, piece_to_index['ep']]), 1)


if __name__ == '__main__':
    unittest.main()
