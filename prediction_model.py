import numpy as np
import chess
from neural_network import fen_to_tensor

def index_to_move(index, board):
    """
    Converts a predicted move index back to a standard chess move in UCI format.

    :param index: The index of the predicted move in the output vector.
    :param board: The current board state as a chess.Board object.
    :return: The move in UCI format (e.g., 'e2e4').
    """

    # Assuming the first 64 indices are for 'from' squares and the next 64 are for 'to' squares
    print(index)
    from_square_index = index % 64
    to_square_index = index // 64

    # Convert square indices to algebraic notation
    from_square = chess.SQUARE_NAMES[from_square_index]
    to_square = chess.SQUARE_NAMES[to_square_index]

    # Combine to get the move in UCI format
    move_uci = from_square + to_square

    # Check if the move is legal
    print(move_uci)
    if move_uci in [move.uci() for move in board.legal_moves]:
        return move_uci
    else:
        return None  # or handle illegal moves differently
