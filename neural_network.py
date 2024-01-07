import pandas as pd
import chess.pgn
import chess
import io
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import numpy as np

piece_to_index = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
        'ck': 12, 'cq': 13, 'CK': 14, 'CQ': 15  # Castling rights
    }

def fen_to_tensor(fen):
    tensor = np.zeros((8, 8, 16), dtype=np.float32)
    parts = fen.split(' ')
    rows = parts[0].split('/')
    for i, row in enumerate(rows):
        # tensor_row = i
        tensor_row = 7 - i
        j = 0
        for char in row:
            if char.isdigit():
                j += int(char)
            else:
                tensor[tensor_row, j, piece_to_index[char]] = 1
                # tensor[tensor_row, j, piece_to_index[char]] = 1
                j += 1
    castling = parts[2]
    if 'K' in castling:
        tensor[:, :, piece_to_index['CK']] = 1
    if 'Q' in castling:
        tensor[:, :, piece_to_index['CQ']] = 1
    if 'k' in castling:
        tensor[:, :, piece_to_index['ck']] = 1
    if 'q' in castling:
        tensor[:, :, piece_to_index['cq']] = 1
    return tensor


def square_to_index(square):
    # Converts a square in algebraic notation to an index (0-63)
    square = square.lower()

    # Validation for the length of the input
    # if len(square) != 2:
    #     raise ValueError("Invalid square length")

    # Validation for file
    file = ord(square[0]) - ord('a')
    if file < 0 or file > 7:
        raise ValueError("Invalid file")

    # Validation for rank
    try:
        rank = 8 - int(square[1])
    except ValueError:
        raise ValueError("Invalid rank")

    if rank < 0 or rank > 7:
        raise ValueError("Invalid rank")

    return 8 * rank + file

def move_to_vector(move):
    # Converts a chess move to a one-hot encoded vector
    # TODO: possibly Check for castling, en passant, pawn promotion
    vector = np.zeros(132, dtype=np.float32)
    from_square = square_to_index(move.uci()[:2])
    to_square = square_to_index(move.uci()[2:4])
    vector[from_square] = 1
    vector[64 + to_square] = 1

    # Handling pawn promotion
    if move.promotion is not None:
        if move.promotion == chess.KNIGHT:
            vector[128] = 1  # Knight promotion
        elif move.promotion == chess.BISHOP:
            vector[129] = 1  # Bishop promotion
        elif move.promotion == chess.ROOK:
            vector[130] = 1  # Rook promotion
        elif move.promotion == chess.QUEEN:
            vector[131] = 1  # Queen promotion

    return vector

def move_sequence_to_vector(move_sequence, max_length=10):
    # Initialize a zero matrix for the sequence with shape (max_length, 128)
    sequence_vector = np.zeros((max_length, 128), dtype=np.float32)

    # Fill the matrix with move vectors
    for i, move in enumerate(move_sequence[-max_length:]):
        from_square = square_to_index(str(move)[:2])
        to_square = square_to_index(str(move)[2:])
        move_vector = np.zeros(128, dtype=np.float32)
        move_vector[from_square] = 1
        move_vector[64 + to_square] = 1
        sequence_vector[i] = move_vector
    return sequence_vector

def move_to_index(move, board):
    from_index = move.from_square
    to_index = move.to_square
    move_index = np.zeros(128, dtype=int)
    move_index[from_index] = 1
    move_index[64 + to_index] = 1
    return move_index

# Parsing the PGN file to extract relevant data
def parse_pgn(file_path, sequence_length=10):
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            recent_moves = []
            for node in game.mainline():
                move = node.move
                fen = board.fen()
                board_tensor = fen_to_tensor(fen)

                recent_moves.append(move)
                if len(recent_moves) > sequence_length:
                    recent_moves.pop(0)

                move_vector = move_to_vector(move)

                # Extracting additional information from comments
                comment = node.comment
                eval_match = re.search(r'\[%eval: ([^\]]+)\]', comment)
                best_move_match = re.search(r'\[%best_move: ([^\]]+)\]', comment)
                played_best_move_match = re.search(r'\[%played_best_move: ([^\]]+)\]', comment)

                evaluation = eval_match.group(1) if eval_match else None
                best_move = best_move_match.group(1) if best_move_match else None
                played_best_move = played_best_move_match.group(1) if played_best_move_match else None
                target_index = move_to_index(board.parse_uci(best_move), board)

                # Yielding the data for each move
                yield board_tensor, recent_moves, move_vector, evaluation, best_move, played_best_move, target_index
                board.push(move)

def create_chess_model():
    # Board input: 8x8x14 tensor
    board_input = Input(shape=(8, 8, 14), name='board_input')

    # CNN layers for board analysis
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(board_input)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Flatten()(x)

    # Move sequence input
    move_input = Input(shape=(10, 128), name='move_input')  # Example: 10 timesteps, each with 128 features
    y = LSTM(64)(move_input)

    # Combine CNN and RNN outputs
    combined = tf.keras.layers.concatenate([x, y])

    # Dense layers for decision making
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)  # Dropout for regularization
    z = Dense(64, activation='relu')(z)

    # Output layer for move prediction
    move_output = Dense(128, activation='softmax', name='move_output')(z)

    # Creating the model
    model = Model(inputs=[board_input, move_input], outputs=[move_output])

    # Compile the model
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Data Generation for Training
def generate_batches(file_path, batch_size=32, sequence_length=10):
    batch_board_tensors = []
    batch_move_sequences = []
    batch_targets = []  # Store the correct moves for each board state

    for board_tensor, recent_moves, move_vector, evaluation, correct_move_vector, played_best_move, target_index  in parse_pgn(file_path):
        move_sequence_vector = move_sequence_to_vector(recent_moves, max_length=sequence_length)

        batch_board_tensors.append(board_tensor)
        batch_move_sequences.append(move_sequence_vector)
        correct_move_vector = target_index
        batch_targets.append(correct_move_vector)  # Assuming you have target data

        if len(batch_board_tensors) == batch_size:
            yield (np.array(batch_board_tensors), np.array(batch_move_sequences)), np.array(batch_targets)
            # yield np.array(batch_board_tensors), np.array(batch_move_sequences), np.array(batch_targets)
            batch_board_tensors = []
            batch_move_sequences = []
            batch_targets = []  # Reset for next batch

for board_tensor, recent_moves, move_vector, evaluation, correct_move_vector, played_best_move, target_index  in parse_pgn('C:/Users/vince/Downloads/games.pgn'):
        tmp = 1
'''
def main():
    # Training Loop
    model = create_chess_model()

    epochs = 10  # Set the number of epochs
    batch_size = 32  # Set the batch size
    train_file_path = 'C:/Users/vince/Downloads/games.pgn'  # Path to training PGN file
    #train_file_path = 'C:/Users/vince/Downloads/GM_games_eval - Copy.pgn'  # Path to training PGN file
    validation_file_path = 'C:/Users/vince/Downloads/validation.pgn'  # Path to validation PGN file
    #validation_file_path = 'C:/Users/vince/Downloads/magnus_evalv2.pgn'  # Path to validation PGN file

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        for (batch_board_tensors, batch_move_sequences), batch_targets in generate_batches(train_file_path, batch_size):
            train_loss, train_accuracy = model.train_on_batch([batch_board_tensors, batch_move_sequences], batch_targets)



        # Validation
        validation_loss, validation_accuracy = 0, 0
        validation_batches = 0
        for batch_board_tensors, batch_move_vectors in generate_batches(validation_file_path, batch_size):
            loss, accuracy = model.test_on_batch(batch_board_tensors, batch_move_vectors)
            validation_loss += loss
            validation_accuracy += accuracy
            validation_batches += 1
    model.summary()
    # model.save('data/grandmaster_model_v1.h5')
    # model.save('data/grandmaster_model_v1.keras')
'''