from tensorflow.keras.models import load_model
import numpy as np
from neural_network import fen_to_tensor, move_sequence_to_vector
from prediction_model import index_to_move
import chess

print("LOADING MODEL!!!!!!!!!!!!!!!!!!!!")
model = load_model('data/grandmaster_model_v1.keras')

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
