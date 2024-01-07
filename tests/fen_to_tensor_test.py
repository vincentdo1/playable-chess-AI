import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'playable-chess-AI')))
from neural_network import fen_to_tensor

initial_position_tensor = np.zeros((8, 8, 16), dtype=np.float32)

# Mapping of pieces to their corresponding indices in the tensor
piece_to_index = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,  # Black pieces
    'R': 9, 'N': 7, 'B': 8, 'Q': 10, 'K': 11, 'P': 6, # White pieces
    'ck': 12, 'cq': 13, 'CK': 14, 'CQ': 15           # Castling rights
}

# Set the pieces on the board
# Black pieces
initial_position_tensor[7, 0, piece_to_index['r']] = 1  # Black rooks
initial_position_tensor[7, 7, piece_to_index['r']] = 1
initial_position_tensor[7, 1, piece_to_index['n']] = 1  # Black knights
initial_position_tensor[7, 6, piece_to_index['n']] = 1
initial_position_tensor[7, 2, piece_to_index['b']] = 1  # Black bishops
initial_position_tensor[7, 5, piece_to_index['b']] = 1
initial_position_tensor[7, 3, piece_to_index['q']] = 1  # Black queen
initial_position_tensor[7, 4, piece_to_index['k']] = 1  # Black king
initial_position_tensor[6, :, piece_to_index['p']] = 1  # Black pawns

# White pieces
initial_position_tensor[0, 0, piece_to_index['R']] = 1  # White rooks
initial_position_tensor[0, 7, piece_to_index['R']] = 1
initial_position_tensor[0, 1, piece_to_index['N']] = 1  # White knights
initial_position_tensor[0, 6, piece_to_index['N']] = 1
initial_position_tensor[0, 2, piece_to_index['B']] = 1  # White bishops
initial_position_tensor[0, 5, piece_to_index['B']] = 1
initial_position_tensor[0, 3, piece_to_index['Q']] = 1  # White queen
initial_position_tensor[0, 4, piece_to_index['K']] = 1  # White king
initial_position_tensor[1, :, piece_to_index['P']] = 1  # White pawns

# Set castling rights (all available at the start)
initial_position_tensor[:, :, piece_to_index['ck']] = 1  # Black kingside
initial_position_tensor[:, :, piece_to_index['cq']] = 1  # Black queenside
initial_position_tensor[:, :, piece_to_index['CK']] = 1  # White kingside
initial_position_tensor[:, :, piece_to_index['CQ']] = 1  # White queenside


specific_position_tensor = np.zeros((8, 8, 16), dtype=np.float32)
specific_position_tensor[4, 3, piece_to_index['q']] = 1

custom_position_tensor = np.zeros((8, 8, 16), dtype=np.float32)

# Black pieces
custom_position_tensor[0, 0, piece_to_index['r']] = 1  # Black rook
custom_position_tensor[0, 2, piece_to_index['b']] = 1  # Black bishop
custom_position_tensor[0, 4, piece_to_index['k']] = 1  # Black king
custom_position_tensor[0, 7, piece_to_index['r']] = 1  # Black rook
custom_position_tensor[1, 1, piece_to_index['p']] = 1  # Black pawn
custom_position_tensor[1, 3, piece_to_index['p']] = 1  # Black pawn
custom_position_tensor[1, 4, piece_to_index['q']] = 1  # Black queen
custom_position_tensor[1, 5, piece_to_index['p']] = 1  # Black pawn
custom_position_tensor[1, 6, piece_to_index['b']] = 1  # Black bishop
custom_position_tensor[2, 2, piece_to_index['n']] = 1  # Black knight
custom_position_tensor[2, 5, piece_to_index['n']] = 1  # Black knight
custom_position_tensor[3, 5, piece_to_index['p']] = 1  # Black pawn
custom_position_tensor[3, 7, piece_to_index['p']] = 1  # Black pawn

# White pieces
custom_position_tensor[4, 2, piece_to_index['B']] = 1  # White bishop
custom_position_tensor[4, 3, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[4, 4, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[5, 2, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[5, 5, piece_to_index['Q']] = 1  # White queen
custom_position_tensor[6, 0, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[6, 1, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[6, 3, piece_to_index['B']] = 1  # White bishop
custom_position_tensor[6, 6, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[6, 7, piece_to_index['P']] = 1  # White pawn
custom_position_tensor[7, 0, piece_to_index['R']] = 1  # White rook
custom_position_tensor[7, 6, piece_to_index['R']] = 1  # White rook
custom_position_tensor[7, 7, piece_to_index['K']] = 1  # White king

# Black's castling rights (kingside and queenside)
custom_position_tensor[:, :, piece_to_index['ck']] = 1
custom_position_tensor[:, :, piece_to_index['cq']] = 1

complex_position_tensor = np.zeros((8, 8, 16), dtype=np.float32)

# Black pieces
complex_position_tensor[7, 0, piece_to_index['r']] = 1  # Black rook
complex_position_tensor[7, 2, piece_to_index['b']] = 1  # Black bishop
complex_position_tensor[7, 4, piece_to_index['k']] = 1  # Black king
complex_position_tensor[7, 7, piece_to_index['r']] = 1  # Black rook
complex_position_tensor[6, 0, piece_to_index['p']] = 1  # Black pawn
complex_position_tensor[6, 1, piece_to_index['p']] = 1  # Black pawn
complex_position_tensor[6, 2, piece_to_index['p']] = 1  # Black pawn
complex_position_tensor[6, 3, piece_to_index['p']] = 1  # Black pawn
complex_position_tensor[6, 4, piece_to_index['q']] = 1  # Black queen
complex_position_tensor[6, 5, piece_to_index['p']] = 1  # Black pawn
complex_position_tensor[6, 6, piece_to_index['b']] = 1  # Black bishop
complex_position_tensor[5, 2, piece_to_index['n']] = 1  # Black knight
complex_position_tensor[5, 5, piece_to_index['n']] = 1  # Black knight
complex_position_tensor[5, 7, piece_to_index['p']] = 1  # Black knight
complex_position_tensor[5, 6, piece_to_index['p']] = 1  # Black knight
complex_position_tensor[4, 4, piece_to_index['p']] = 1  # Black pawn

# White pieces
complex_position_tensor[3, 2, piece_to_index['B']] = 1  # White bishop
complex_position_tensor[3, 3, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[3, 4, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[2, 2, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[2, 5, piece_to_index['Q']] = 1  # White queen
complex_position_tensor[1, 0, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[1, 1, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[1, 5, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[1, 2, piece_to_index['B']] = 1  # White bishop
complex_position_tensor[1, 6, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[1, 7, piece_to_index['P']] = 1  # White pawn
complex_position_tensor[0, 0, piece_to_index['R']] = 1  # White rook
complex_position_tensor[0, 5, piece_to_index['R']] = 1  # White rook
complex_position_tensor[0, 6, piece_to_index['K']] = 1  # White king

# Black's castling rights (kingside and queenside)
complex_position_tensor[:, :, piece_to_index['ck']] = 1
complex_position_tensor[:, :, piece_to_index['cq']] = 1

def tensor_to_fen_corrected(tensor):
    # Reverse mapping from indices to chess pieces
    index_to_piece = {
        0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k',  # Black pieces
        6: 'P', 7: 'N', 8: 'B', 9: 'R', 10: 'Q', 11: 'K'  # White pieces
    }
    
    # Initialize an empty board
    board = [['' for _ in range(8)] for _ in range(8)]
    
    # Populate the board with pieces based on the tensor
    for row in range(8):
        for col in range(8):
            for piece_index in range(12):  # Only iterate over piece layers
                if tensor[7 - row, col, piece_index] == 1:
                    board[row][col] = index_to_piece[piece_index]
                    break
    
    # Construct the board part of the FEN string
    fen_rows = []
    for row in board:
        empty = 0
        fen_row = ''
        for cell in row:
            if cell == '':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    fen_board = '/'.join(fen_rows)

    # Construct the castling part of the FEN string
    castling = ''
    if tensor[0, 0, 14] == 1:  # White kingside
        castling += 'K'
    if tensor[0, 0, 15] == 1:  # White queenside
        castling += 'Q'
    if tensor[0, 0, 12] == 1:  # Black kingside
        castling += 'k'
    if tensor[0, 0, 13] == 1:  # Black queenside
        castling += 'q'
    castling = castling or '-'

    # Active player, half-move clock, and full-move number are not available in the tensor
    active_player = 'w'  # Default active player
    half_move_clock = '0'  # Default half-move clock
    full_move_number = '1'  # Default full-move number

    # Construct the full FEN string
    fen = f"{fen_board} {active_player} {castling} - {half_move_clock} {full_move_number}"
    return fen
    
class TestFenToTensor(unittest.TestCase):

    def test_starting_position(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
        expected_output = initial_position_tensor
        print(tensor_to_fen_corrected(expected_output))
        tensor = fen_to_tensor(fen)
        print(tensor_to_fen_corrected(tensor))
        np.testing.assert_array_equal(tensor, expected_output)

    def test_custom_position(self):
        fen = "8/8/8/3q4/8/8/8/8 w - - 0 1"
        expected_output = specific_position_tensor
        tensor = fen_to_tensor(fen)
        print(tensor_to_fen_corrected(expected_output))
        print(tensor_to_fen_corrected(tensor))
        np.testing.assert_array_equal(tensor, expected_output)

    def test_complex_position(self):
        fen = "r1b1k2r/ppppqpb1/2n2npp/4p3/2BPP3/2P2Q2/PPB2PPP/R4RK1 w kq - 0 1"
        expected_output = complex_position_tensor
        tensor = fen_to_tensor(fen)
        print(tensor_to_fen_corrected(expected_output))
        print(tensor_to_fen_corrected(tensor))
        np.testing.assert_array_equal(tensor, expected_output)

if __name__ == '__main__':
    unittest.main()
