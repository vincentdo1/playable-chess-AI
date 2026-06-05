"""Piece-square evaluation tables with endgame awareness."""

import chess

# The piece-square / positional tables below were tuned ~10x larger than the
# pawn-unit material values (pawn=1, knight=3.2, ...). Left unscaled they
# outweigh material, so the search trades pieces for placement. Scale every
# positional term down to keep material the dominant factor.
PST_SCALE = 0.1

# Middlegame piece-square tables

pawnEvalWhite = (
    (0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    (8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0,  8.0),
    (2.0,  2.0,  3.0,  5.0,  5.0,  3.0,  2.0,  2.0),
    (0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5),
    (0.0,  0.0,  0.5,  2.0,  2.0,  0.5,  0.0,  0.0),
    (0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5),
    (0.5,  1.0,  0.5, -2.0, -2.0,  0.5,  1.0,  0.5),
    (0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
)
pawnEvalBlack = tuple(reversed(pawnEvalWhite))

knightEval = (
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
    (-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0),
    (-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0),
    (-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0),
    (-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0),
    (-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0),
    (-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0),
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
)

bishopEvalWhite = (
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
    (-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0),
    (-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0),
    (-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0),
    (-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0),
    (-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0),
    (-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0),
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
)
bishopEvalBlack = tuple(reversed(bishopEvalWhite))

rookEvalWhite = (
    ( 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    ( 0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5),
    (-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5),
    (-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5),
    (-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5),
    (-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5),
    (-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5),
    ( 0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0),
)
rookEvalBlack = tuple(reversed(rookEvalWhite))

queenEval = (
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
    (-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0),
    (-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0),
    (-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5),
    ( 0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5),
    (-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0),
    (-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0),
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
)

# Middlegame king — stay safe, behind pawns
kingEvalWhite = (
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0),
    (-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0),
    ( 2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0),
    ( 2.0,  3.0,  3.0,  0.0,  0.0,  1.0,  3.0,  2.0),
)
kingEvalBlack = tuple(reversed(kingEvalWhite))

# Endgame king table — king should centralize and chase pawns
# Positive values in the center, penalize corners and edges
kingEndgameEval = (
    (-5.0, -4.0, -3.0, -2.0, -2.0, -3.0, -4.0, -5.0),
    (-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0),
    (-3.0,  0.0,  2.0,  3.0,  3.0,  2.0,  0.0, -3.0),
    (-2.0,  0.0,  3.0,  4.0,  4.0,  3.0,  0.0, -2.0),
    (-2.0,  0.0,  3.0,  4.0,  4.0,  3.0,  0.0, -2.0),
    (-3.0,  0.0,  2.0,  3.0,  3.0,  2.0,  0.0, -3.0),
    (-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0),
    (-5.0, -4.0, -3.0, -2.0, -2.0, -3.0, -4.0, -5.0),
)

# Endgame detection

def is_endgame(board: chess.Board) -> bool:
    """True when queens are gone or material is very low — triggers endgame evaluation."""
    white_queens = len(board.pieces(chess.QUEEN,  chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN,  chess.BLACK))
    white_rooks  = len(board.pieces(chess.ROOK,   chess.WHITE))
    black_rooks  = len(board.pieces(chess.ROOK,   chess.BLACK))
    white_minor  = (len(board.pieces(chess.BISHOP, chess.WHITE)) +
                    len(board.pieces(chess.KNIGHT, chess.WHITE)))
    black_minor  = (len(board.pieces(chess.BISHOP, chess.BLACK)) +
                    len(board.pieces(chess.KNIGHT, chess.BLACK)))

    # No queens at all
    if white_queens == 0 and black_queens == 0:
        return True

    # Both have a queen but no rooks and very little minor material
    if (white_queens <= 1 and black_queens <= 1 and
            white_rooks == 0 and black_rooks == 0 and
            white_minor <= 1 and black_minor <= 1):
        return True

    return False

def get_search_depth(board: chess.Board) -> int:
    """Deeper search in endgame (fewer pieces = smaller tree)."""
    pieces = len(board.piece_map())
    if pieces > 20:
        return 2   # opening/middlegame — stay fast
    elif pieces > 10:
        return 3   # middlegame transition
    else:
        return 4   # endgame — few pieces, deeper is cheap

# Passed pawn helpers

def _is_passed_pawn(board: chess.Board, square: chess.Square,
                    color: chess.Color) -> bool:
    """True if no opposing pawn can block or capture this pawn on its path to promotion."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    files_to_check = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]

    if color == chess.WHITE:
        ranks_ahead = range(rank + 1, 8)
        opponent    = chess.BLACK
    else:
        ranks_ahead = range(rank - 1, -1, -1)
        opponent    = chess.WHITE

    for r in ranks_ahead:
        for f in files_to_check:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == opponent:
                return False
    return True

def _passed_pawn_bonus(rank: int, color: chess.Color) -> float:
    """Bonus that scales with how far advanced the passed pawn is."""
    if color == chess.WHITE:
        advance = rank  # rank 6 = one step from promotion
    else:
        advance = 7 - rank
    bonuses = (0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 0.0)
    return bonuses[advance]

def _king_pawn_proximity(king_sq: chess.Square,
                         pawn_sq: chess.Square) -> float:
    """Chebyshev distance bonus — reward king for being close to a passed pawn."""
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
    dist   = max(abs(kf - pf), abs(kr - pr))
    return max(0.0, 4.0 - dist)  # 4.0 adjacent, 3.0 one away, ...

# Main evaluation function

def evaluate(board_array, chess_board: chess.Board = None) -> float:
    """Score from White's perspective. Pass chess_board to enable endgame features."""
    endgame = is_endgame(chess_board) if chess_board is not None else False
    score   = 0.0

    for x, y, piece in board_array[0]:
        if piece == "p":
            score += 1 + PST_SCALE * pawnEvalWhite[y - 1][x - 1]
        elif piece == "b":
            score += 3.3 + PST_SCALE * bishopEvalWhite[y - 1][x - 1]     
        elif piece == "n":
            score += 3.2 + PST_SCALE * knightEval[y - 1][x - 1]           
        elif piece == "r":
            score += 5.0 + PST_SCALE * rookEvalWhite[y - 1][x - 1]         
        elif piece == "q":
            score += 9.0 + PST_SCALE * queenEval[y - 1][x - 1] 
        elif piece == "k":
            score += 200
            if endgame:
                score += PST_SCALE * kingEndgameEval[y - 1][x - 1]
            else:
                score += PST_SCALE * kingEvalWhite[y - 1][x - 1]

    for x, y, piece in board_array[1]:
        if piece == "p":
            score -= 1 + PST_SCALE * pawnEvalBlack[y - 1][x - 1]
        elif piece == "b":
            score -= 3.3 + PST_SCALE * bishopEvalBlack[y - 1][x - 1]
        elif piece == "n":
            score -= 3.2 + PST_SCALE * knightEval[y - 1][x - 1]
        elif piece == "r":
            score -= 5.0 + PST_SCALE * rookEvalBlack[y - 1][x - 1]
        elif piece == "q":
            score -= 9.0 + PST_SCALE * queenEval[y - 1][x - 1]
        elif piece == "k":
            score -= 200
            if endgame:
                score -= PST_SCALE * kingEndgameEval[y - 1][x - 1]
            else:
                score -= PST_SCALE * kingEvalBlack[y - 1][x - 1]

    if endgame and chess_board is not None:

        white_king_sq = chess_board.king(chess.WHITE)
        black_king_sq = chess_board.king(chess.BLACK)

        for sq in chess_board.pieces(chess.PAWN, chess.WHITE):
            if _is_passed_pawn(chess_board, sq, chess.WHITE):
                rank  = chess.square_rank(sq)
                score += PST_SCALE * _passed_pawn_bonus(rank, chess.WHITE)
                if white_king_sq is not None:
                    score += PST_SCALE * 0.3 * _king_pawn_proximity(white_king_sq, sq)
                if black_king_sq is not None:
                    score -= PST_SCALE * 0.2 * _king_pawn_proximity(black_king_sq, sq)

        for sq in chess_board.pieces(chess.PAWN, chess.BLACK):
            if _is_passed_pawn(chess_board, sq, chess.BLACK):
                rank  = chess.square_rank(sq)
                score -= PST_SCALE * _passed_pawn_bonus(rank, chess.BLACK)
                if black_king_sq is not None:
                    score -= PST_SCALE * 0.3 * _king_pawn_proximity(black_king_sq, sq)
                if white_king_sq is not None:
                    score += PST_SCALE * 0.2 * _king_pawn_proximity(white_king_sq, sq)

        only_kings_and_pawns = all(
            p.piece_type in (chess.KING, chess.PAWN)
            for p in chess_board.piece_map().values()
        )
        if only_kings_and_pawns:
            white_pawns = len(chess_board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = len(chess_board.pieces(chess.PAWN, chess.BLACK))
            score += PST_SCALE * 1.5 * (white_pawns - black_pawns)

    return score