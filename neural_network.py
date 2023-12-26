import pandas as pd
import chess.pgn
import chess
import io
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, LSTM
from tensorflow.keras.models import Model
import numpy as np

piece_to_index = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
}

def fen_to_tensor(fen):
    # Initialize an empty tensor
    tensor = np.zeros((8, 8, 14), dtype=np.float32)
    
    # Split the FEN string to get the piece placement
    pieces = fen.split(' ')[0]
    rows = pieces.split('/')
    
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                j += int(char)
            else:
                # Set the corresponding layer for the piece
                tensor[i, j, piece_to_index[char]] = 1
                j += 1
    return tensor

def square_to_index(square):
    # Converts a square in algebraic notation to an index (0-63)
    rank = 8 - int(square[1]) # Rank (row)
    file = ord(square[0]) - ord('a') # File (column)
    return 8 * rank + file

def move_to_vector(move):
    # Converts a chess move to a one-hot encoded vector
    vector = np.zeros(128, dtype=np.float32)
    from_square = square_to_index(str(move)[:2])
    to_square = square_to_index(str(move)[2:])
    vector[from_square] = 1
    vector[64 + to_square] = 1
    return vector


# Parsing the PGN file to extract relevant data
def parse_pgn(file_path):
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            for node in game.mainline():
                move = node.move
                fen = board.fen()
                board_tensor = fen_to_tensor(fen)
                move_vector = move_to_vector(move)

                # Extracting additional information from comments
                comment = node.comment
                eval_match = re.search(r'\[%eval: ([^\]]+)\]', comment)
                best_move_match = re.search(r'\[%best_move: ([^\]]+)\]', comment)
                played_best_move_match = re.search(r'\[%played_best_move: ([^\]]+)\]', comment)

                evaluation = eval_match.group(1) if eval_match else None
                best_move = best_move_match.group(1) if best_move_match else None
                played_best_move = played_best_move_match.group(1) if played_best_move_match else None

                # Yielding the data for each move
                yield board_tensor, move_vector, evaluation, best_move, played_best_move
                board.push(move)

def create_chess_model():
    # Board input: 8x8x14 tensor
    board_input = Input(shape=(8, 8, 14), name='board_input')

    # CNN layers
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(board_input)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)

    # Move sequence input: This should be a sequence of move vectors
    # Adjust the input shape as per your move encoding
    move_input = Input(shape=(None, 128), name='move_input')

    # RNN layer
    y = LSTM(64)(move_input)

    # Combine CNN and RNN outputs
    combined = tf.keras.layers.concatenate([x, y])

    # Dense layers for decision making
    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)

    # Output layer: Adjust the number of units to match the size of your move encoding
    move_output = Dense(128, activation='softmax', name='move_output')(z)

    model = Model(inputs=[board_input, move_input], outputs=[move_output])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Use this function to parse your PGN file
for board_tensor, move_vector, evaluation, best_move, played_best_move in parse_pgn('C:/Users/vince/Downloads/games.pgn'):
    print(board_tensor, move_vector, evaluation, best_move, played_best_move)

chess_model = create_chess_model()
chess_model.summary()