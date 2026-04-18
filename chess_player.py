import os
import math
import random
import chess
import chess.engine
import heuristics

STOCKFISH_PATH = os.environ.get(
    'STOCKFISH_PATH',
    'stockfish.exe'
)

def evaluate_helper(board):
    """Convert board to [[white_pieces], [black_pieces]] for evaluate()."""
    piece_map    = board.piece_map()
    white_pieces = []
    black_pieces = []
    for square, piece in piece_map.items():
        piece_type = piece.symbol().lower()
        x = chess.square_file(square)
        y = chess.square_rank(square)
        if piece.color == chess.WHITE:
            white_pieces.append((x + 1, 8 - y, piece_type))
        else:
            black_pieces.append((x + 1, 8 - y, piece_type))
    return [white_pieces, black_pieces]

def get_best_move(board, time_limit=2):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

def random_move_player(board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)

def alphabeta(side, board, depth=None, alpha=-math.inf, beta=math.inf):
    """
    Top-level alphabeta call. If depth is None, uses adaptive depth
    from heuristics.get_search_depth() based on game phase.
    """
    if depth is None:
        depth = heuristics.get_search_depth(board)

    best_move = None
    max_eval  = float("-inf") if side else float("inf")
    alpha     = float("-inf")
    beta      = float("inf")

    for move in board.legal_moves:
        board.push(move)
        evaluation = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
        board.pop()
        if side and evaluation > max_eval:
            max_eval  = evaluation
            best_move = move
        elif not side and evaluation < max_eval:
            max_eval  = evaluation
            best_move = move

    return best_move

def alphabetahelper(side, board, depth, alpha, beta):
    if depth == 0 or board.is_game_over() or board.is_checkmate():
        # Pass the chess.Board to evaluate() so endgame features activate
        return heuristics.evaluate(evaluate_helper(board), board)

    if side:
        value = -10000
        for move in board.legal_moves:
            board.push(move)
            v = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
            board.pop()
            value = max(value, v)
            if value >= beta:
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
            if value <= alpha:
                break
            beta = min(beta, value)
        return value