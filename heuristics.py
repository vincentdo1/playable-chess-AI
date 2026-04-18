"""
heuristics.py — Piece-square evaluation tables.

"""

import chess

# ---------------------------------------------------------------------------
# Middlegame piece-square tables
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Endgame king table — king should centralize and chase pawns
# Positive values in the center, penalize corners and edges
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Endgame detection
# ---------------------------------------------------------------------------

def is_endgame(board: chess.Board) -> bool:
    """
    Detect endgame phase. True when:
      - Both sides have no queens, OR
      - Both sides have queens but very little other material (rooks gone)

    This triggers the endgame king table and passed pawn bonuses.
    """
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
    """
    Adaptive search depth based on game phase.
    Endgame trees are much smaller so we can search deeper for free.
    """
    pieces = len(board.piece_map())
    if pieces > 20:
        return 2   # opening/middlegame — stay fast
    elif pieces > 10:
        return 3   # middlegame transition
    else:
        return 4   # endgame — few pieces, deeper is cheap


# ---------------------------------------------------------------------------
# Passed pawn helpers
# ---------------------------------------------------------------------------

def _is_passed_pawn(board: chess.Board, square: chess.Square,
                    color: chess.Color) -> bool:
    """
    A pawn is passed if no opposing pawn can ever block or capture it —
    i.e. no opposing pawn on the same file or adjacent files ahead of it.
    """
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
    """
    Bonus for a passed pawn based on how advanced it is.
    The closer to promotion, the more valuable.
    """
    if color == chess.WHITE:
        advance = rank  # rank 6 = one step from promotion
    else:
        advance = 7 - rank
    # Scale: rank 1=0.5, rank 2=1.0, rank 3=2.0, rank 4=3.5, rank 5=5.0, rank 6=8.0
    bonuses = (0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 0.0)
    return bonuses[advance]


def _king_pawn_proximity(king_sq: chess.Square,
                         pawn_sq: chess.Square) -> float:
    """
    Bonus for the king being close to a passed pawn in the endgame.
    Chebyshev distance — king can move diagonally.
    Max bonus when adjacent, zero when far away.
    """
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
    dist   = max(abs(kf - pf), abs(kr - pr))
    return max(0.0, 4.0 - dist)  # 4.0 adjacent, 3.0 one away, ...


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(board_array, chess_board: chess.Board = None) -> float:
    """
    Evaluate the board position. Returns a score from White's perspective
    (positive = good for White, negative = good for Black).

    Args:
        board_array  : [[white_pieces], [black_pieces]] in (x, y, type) format
                       as produced by chess_player.evaluate_helper()
        chess_board  : optional chess.Board for endgame features.
                       If None, uses middlegame tables only (backward compatible).
    """
    endgame = is_endgame(chess_board) if chess_board is not None else False
    score   = 0.0

    # --- White pieces ---
    for x, y, piece in board_array[0]:
        if piece == "p":
            score += 1 + pawnEvalWhite[y - 1][x - 1]
        elif piece == "b":
            score += 9 + bishopEvalWhite[y - 1][x - 1]
        elif piece == "n":
            score += 9 + knightEval[y - 1][x - 1]
        elif piece == "r":
            score += 14 + rookEvalWhite[y - 1][x - 1]
        elif piece == "q":
            score += 25 + queenEval[y - 1][x - 1]
        elif piece == "k":
            score += 200
            if endgame:
                score += kingEndgameEval[y - 1][x - 1]
            else:
                score += kingEvalWhite[y - 1][x - 1]

    # --- Black pieces ---
    for x, y, piece in board_array[1]:
        if piece == "p":
            score -= 1 + pawnEvalBlack[y - 1][x - 1]
        elif piece == "b":
            score -= 9 + bishopEvalBlack[y - 1][x - 1]
        elif piece == "n":
            score -= 9 + knightEval[y - 1][x - 1]
        elif piece == "r":
            score -= 14 + rookEvalBlack[y - 1][x - 1]
        elif piece == "q":
            score -= 25 + queenEval[y - 1][x - 1]
        elif piece == "k":
            score -= 200
            if endgame:
                score -= kingEndgameEval[y - 1][x - 1]
            else:
                score -= kingEvalBlack[y - 1][x - 1]

    # --- Endgame bonus features ---
    if endgame and chess_board is not None:

        white_king_sq = chess_board.king(chess.WHITE)
        black_king_sq = chess_board.king(chess.BLACK)

        # Passed pawn bonuses + king proximity
        for sq in chess_board.pieces(chess.PAWN, chess.WHITE):
            if _is_passed_pawn(chess_board, sq, chess.WHITE):
                rank  = chess.square_rank(sq)
                score += _passed_pawn_bonus(rank, chess.WHITE)
                # Bonus for own king supporting the passer
                if white_king_sq is not None:
                    score += 0.3 * _king_pawn_proximity(white_king_sq, sq)
                # Penalty when enemy king is close to our passer (blocking it)
                if black_king_sq is not None:
                    score -= 0.2 * _king_pawn_proximity(black_king_sq, sq)

        for sq in chess_board.pieces(chess.PAWN, chess.BLACK):
            if _is_passed_pawn(chess_board, sq, chess.BLACK):
                rank  = chess.square_rank(sq)
                score -= _passed_pawn_bonus(rank, chess.BLACK)
                if black_king_sq is not None:
                    score -= 0.3 * _king_pawn_proximity(black_king_sq, sq)
                if white_king_sq is not None:
                    score += 0.2 * _king_pawn_proximity(white_king_sq, sq)

        # Bonus for having more pawns in pure king+pawn endgames
        # (material advantage is more decisive when there's nothing else)
        only_kings_and_pawns = all(
            p.piece_type in (chess.KING, chess.PAWN)
            for p in chess_board.piece_map().values()
        )
        if only_kings_and_pawns:
            white_pawns = len(chess_board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = len(chess_board.pieces(chess.PAWN, chess.BLACK))
            score += 1.5 * (white_pawns - black_pawns)

    return score