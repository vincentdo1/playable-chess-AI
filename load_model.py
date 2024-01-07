from tensorflow.keras.models import load_model
import numpy as np
from neural_network import fen_to_tensor, move_sequence_to_vector
from prediction_model import index_to_move
import chess

print("LOADING MODEL!!!!!!!!!!!!!!!!!!!!")
model = load_model('data/grandmaster_model_v1.keras')

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

def predict_next_move(model, board):
    board_tensor = fen_to_tensor(board.fen())
    move_sequence_vector = move_sequence_to_vector([move.uci() for move in board.move_stack[-10:]], max_length=10)
    print(move_sequence_vector)
    predicted_move_prob = model.predict([np.array([board_tensor]), np.array([move_sequence_vector])])
    print(predicted_move_prob)
    predicted_move_index = np.argmax(predicted_move_prob[0])
    predicted_move = index_to_move(predicted_move_index, board)
    return predicted_move

board = chess.Board()
next_move = predict_next_move(model, board)
print("Predicted next move:", next_move)
