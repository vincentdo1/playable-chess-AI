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

_PIECE_VALUES_CP = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN:  900, chess.KING:     0,
}

def _order_moves(board):
    scored = []
    for move in board.legal_moves:
        score = 0
        if board.is_capture(move):
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_val   = _PIECE_VALUES_CP.get(victim.piece_type,   0) if victim   else 0
            attacker_val = _PIECE_VALUES_CP.get(attacker.piece_type, 0) if attacker else 0
            score = 10000 + victim_val - attacker_val

        # Queen promotion is the highest priority move — beats any check, fix in next update
        if move.promotion == chess.QUEEN:
            score += 11000
        elif move.promotion:
            score += 6000   # underpromotions still useful, but low priority

        if board.gives_check(move):
            score += 9000

        scored.append((score, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in scored]

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

def alphabeta(side, board, depth=None):
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

    for move in _order_moves(board):
        board.push(move)
        evaluation = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
        board.pop()
        if side and evaluation > max_eval:
            max_eval  = evaluation
            best_move = move
            alpha = max(alpha, max_eval)
        elif not side and evaluation < max_eval:
            max_eval  = evaluation
            best_move = move
            beta = min(beta, max_eval)

    return best_move

def alphabetahelper(side, board, depth, alpha, beta):
    if board.is_checkmate():
        # board.turn is the side being mated
        if board.turn == chess.WHITE:
            return -(10000 - depth)   # White is mated → bad for White
        else:
            return 10000 - depth      # Black is mated → great for White
    if board.is_repetition(3):
        return 0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return heuristics.evaluate(evaluate_helper(board), board)

    if side:
        value = -10000
        for move in _order_moves(board):
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
        for move in _order_moves(board):
            board.push(move)
            v = alphabetahelper(board.turn, board, depth - 1, alpha, beta)
            board.pop()
            value = min(value, v)
            if value <= alpha:
                break
            beta = min(beta, value)
        return value