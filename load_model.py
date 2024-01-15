from tensorflow.keras.models import load_model
import numpy as np
from neural_network import fen_to_tensor, move_sequence_to_vector
import chess

print("LOADING MODEL!!!!!!!!!!!!!!!!!!!!")
model = load_model('data/grandmaster_model_v2.keras')
# model = load_model('data/tmp.keras')

def index_to_move(predicted_index, board):
    if predicted_index < 64:
        # The index represents only a 'from' square, which is not sufficient to determine the move.
        return None
    elif predicted_index < 128:
        # Normal move (not a promotion). The index can be broken down into 'from' and 'to' squares.
        from_square_index = predicted_index % 64
        to_square_index = predicted_index // 64
        from_square = chess.SQUARE_NAMES[from_square_index]
        to_square = chess.SQUARE_NAMES[to_square_index]
        move_uci = from_square + to_square
    else:
        # Pawn promotion. Additional logic is needed here to determine the correct move.
        return None  # Placeholder for promotion logic

    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        return move_uci
    else:
        return None


def predict_next_move(model, board):
    board_tensor = fen_to_tensor(board.fen())
    move_sequence_vector = move_sequence_to_vector([move.uci() for move in board.move_stack[-10:]], max_length=10)
    print(move_sequence_vector)
    predicted_move_prob = model.predict([np.array([board_tensor]), np.array([move_sequence_vector])])
    print(predicted_move_prob)
    predicted_move_index = np.argmax(predicted_move_prob[0])
    print(predicted_move_index)
    predicted_move = index_to_move(predicted_move_index, board)
    return predicted_move

board = chess.Board()
next_move = predict_next_move(model, board)
print("Predicted next move:", next_move)
