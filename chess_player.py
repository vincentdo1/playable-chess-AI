import math
import random
import pygame
import chess
import chess.pgn
import chess.svg
import chess.engine
import heuristics

# This function converts the board format into an array for the evaluate function in mychess.lib.heuristics.
def evaluate_helper(board):
    piece_map = board.piece_map()
    white_pieces = []
    black_pieces = []

    for square, piece in piece_map.items():
        piece_type = piece.symbol().lower()
        x, y = chess.square_file(square), chess.square_rank(square)

        if piece.color == chess.WHITE:
            white_pieces.append((x + 1, 8 - y, piece_type))
        else:
            black_pieces.append((x + 1, 8 - y, piece_type))
    return [white_pieces] + [black_pieces]

def get_best_move(board, time_limit=2):
    with chess.engine.SimpleEngine.popen_uci("C:/Users/Vincent/Downloads/scid_windows_5.0.2/scid_windows_x64/engines/stockfish.exe") as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        engine.close()
        return result.move

def random_move_player(board):
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None  # If no legal moves available, return None or any other indication of inability to move

    random_move = random.choice(legal_moves)
    return random_move
    
def alphabeta(side, board, depth, alpha=-math.inf, beta=math.inf):
    best_move = None
    max_eval = float("-inf") if side else float("inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in board.legal_moves:
        board.push(move)
        evaluation = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
        board.pop()
        if side and evaluation > max_eval:
            max_eval = evaluation
            best_move = move
        elif not side and evaluation < max_eval:
            max_eval = evaluation
            best_move = move

    return best_move

def alphabetahelper(side, board, depth, alpha=-math.inf, beta=math.inf):
    if depth == 0 or board.is_game_over() or board.is_checkmate():
      return heuristics.evaluate(evaluate_helper(board))
    if (side):
      value = -10000
      for move in board.legal_moves:
        board.push(move)
        v = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
        board.pop()
        value = max(value, v)
        if (value >= beta):
          break
        alpha = max(alpha, value)
      return value
    else:
      value = 10000
      for move in board.legal_moves:
        board.push(move)
        v = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
        board.pop()
        value = min(value, v)
        if (value <= alpha):
          break
        beta = min(beta, value)
      return value