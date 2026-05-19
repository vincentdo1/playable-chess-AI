"""Converts PGN files to .npz chunks for fast training. Run once before training.

Training labels come from the move actually played in the PGN. Optional
Stockfish annotations are kept only for analysis, such as measuring how often
the played move matched the engine's top move.
"""

import io
import argparse
import glob
import os
import re
import zipfile
import numpy as np
import chess
import chess.engine
import chess.pgn

import chess_player
from neural_network import (
    fen_to_tensor, legal_policy_indices, move_sequence_to_vector,
    move_to_policy_index
)

# Config. Environment variables can override these defaults.
GM_ZIP = os.environ.get(
    'GM_ZIP', os.path.join('extractions', 'GM_games_2600.zip')
)
MAGNUS_ZIP = os.environ.get(
    'MAGNUS_ZIP', os.path.join('extractions', 'magnus.zip')
)

GM_EVAL_PGN = os.environ.get(
    'GM_EVAL_PGN', 'GM_games_eval.pgn'
)  # name of the eval file inside GM_ZIP
MAGNUS_EVAL_PGN = os.environ.get(
    'MAGNUS_EVAL_PGN', 'magnus_eval.pgn'
)  # name of the eval file inside MAGNUS_ZIP

TRAIN_DIR = os.environ.get('TRAIN_DIR', 'data/train_chunks')
VAL_DIR = os.environ.get('VAL_DIR', 'data/val_chunks')
TEST_DIR = os.environ.get('TEST_DIR', 'data/test_chunks')

CHUNK_SIZE       = 50_000  # positions per .npz file
SEQ_LEN          = 10
TRAIN_SPLIT      = float(os.environ.get('TRAIN_SPLIT', '0.80'))
VAL_SPLIT        = float(os.environ.get('VAL_SPLIT', '0.10'))
TEST_SPLIT       = float(os.environ.get('TEST_SPLIT', '0.10'))
ALLOW_EXISTING_CHUNKS = os.environ.get(
    'ALLOW_EXISTING_CHUNKS', '0'
).lower() in {'1', 'true', 'yes'}

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', 'stockfish.exe')
CALCULATE_CP_LOSS = os.environ.get('CALCULATE_CP_LOSS', '1').lower() not in {
    '0', 'false', 'no'
}
CP_LOSS_TIME_LIMIT = float(os.environ.get('CP_LOSS_TIME_LIMIT', '0.05'))
CP_LOSS_DEPTH = os.environ.get('CP_LOSS_DEPTH')
CP_LOSS_DEPTH = int(CP_LOSS_DEPTH) if CP_LOSS_DEPTH else None
MATE_SCORE_CP = 100_000
VALUE_CP_SCALE = float(os.environ.get('VALUE_CP_SCALE', '600'))

NO_ENGINE_ANNOTATION = -1
ENGINE_BEST_NOT_PLAYED = 0
ENGINE_BEST_PLAYED = 1
UNKNOWN_CP_LOSS = -1.0
VALUE_FROM_RESULT = 0
VALUE_FROM_EVAL = 1
POLICY_TARGET_POSITIVE = 1
POLICY_TARGET_NEGATIVE = -1
BAD_MOVE_CP_LOSS_THRESHOLD = float(
    os.environ.get('BAD_MOVE_CP_LOSS_THRESHOLD', '150')
)
INACCURACY_CP_LOSS_THRESHOLD = float(
    os.environ.get('INACCURACY_CP_LOSS_THRESHOLD', '50')
)
MISTAKE_CP_LOSS_THRESHOLD = float(
    os.environ.get('MISTAKE_CP_LOSS_THRESHOLD', str(BAD_MOVE_CP_LOSS_THRESHOLD))
)
BLUNDER_CP_LOSS_THRESHOLD = float(
    os.environ.get('BLUNDER_CP_LOSS_THRESHOLD', '300')
)
CRITICAL_CP_LOSS_THRESHOLD = float(
    os.environ.get('CRITICAL_CP_LOSS_THRESHOLD', '900')
)
INCLUDE_BAD_MOVE_NEGATIVES = os.environ.get(
    'INCLUDE_BAD_MOVE_NEGATIVES', '1'
).lower() not in {'0', 'false', 'no'}
TAGGED_POSITIVE_CP_LOSS = os.environ.get(
    'TAGGED_POSITIVE_CP_LOSS', '0'
).lower() in {'1', 'true', 'yes'}
SEARCH_NEGATIVE_CANDIDATES = int(
    os.environ.get('SEARCH_NEGATIVE_CANDIDATES', '0')
)
SEARCH_NEGATIVE_MAX_PER_POSITION = int(
    os.environ.get('SEARCH_NEGATIVE_MAX_PER_POSITION', '2')
)
SEARCH_NEGATIVE_DEPTH = int(os.environ.get('SEARCH_NEGATIVE_DEPTH', '2'))
SEARCH_NEGATIVE_THRESHOLD = float(
    os.environ.get('SEARCH_NEGATIVE_THRESHOLD', '250')
)
SEARCH_NEGATIVE_ONLY = os.environ.get(
    'SEARCH_NEGATIVE_ONLY', '0'
).lower() in {'1', 'true', 'yes'}
CP_LOSS_BUCKET_UNKNOWN = -1
CP_LOSS_BUCKET_OK = 0
CP_LOSS_BUCKET_INACCURACY = 1
CP_LOSS_BUCKET_MISTAKE = 2
CP_LOSS_BUCKET_BLUNDER = 3
CP_LOSS_BUCKET_CRITICAL = 4

_PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

def _engine_limit():
    if CP_LOSS_DEPTH is not None:
        return chess.engine.Limit(depth=CP_LOSS_DEPTH)
    return chess.engine.Limit(time=CP_LOSS_TIME_LIMIT)

def _score_to_cp(score, pov_color):
    return score.pov(pov_color).score(mate_score=MATE_SCORE_CP)

def _parse_eval_annotation(comment, pov_color):
    """Parse this project's [%eval: ...] annotation into centipawns."""
    eval_match = re.search(r'\[%eval:?\s+([^\]]+)\]', comment)
    if not eval_match:
        return None

    value = eval_match.group(1).strip()
    try:
        if value.startswith('#'):
            mate = int(value[1:])
            white_cp = MATE_SCORE_CP if mate > 0 else -MATE_SCORE_CP
        else:
            numeric = float(value)
            white_cp = int(round(numeric * 100)) if '.' in value else int(numeric)
    except ValueError:
        return None

    return white_cp if pov_color == chess.WHITE else -white_cp

def _position_eval_cp(board, comment, engine):
    """Evaluate the current position from the side-to-move's perspective."""
    pov_color = board.turn
    eval_cp = _parse_eval_annotation(comment, pov_color)
    if eval_cp is not None:
        return eval_cp
    if engine is None:
        return None
    info = engine.analyse(board, _engine_limit())
    return _score_to_cp(info['score'], pov_color)

def _cp_to_value(cp):
    """Compress centipawns into a bounded [-1, 1] value target."""
    return float(np.tanh(float(cp) / VALUE_CP_SCALE))

def _result_value_for_color(result, color):
    """Return the final game result from the requested color's perspective."""
    if result == '1-0':
        return 1.0 if color == chess.WHITE else -1.0
    if result == '0-1':
        return -1.0 if color == chess.WHITE else 1.0
    if result == '1/2-1/2':
        return 0.0
    return 0.0

def _sample_weight_from_cp_loss(cp_loss):
    if cp_loss == UNKNOWN_CP_LOSS:
        return 1.0
    if cp_loss <= 25:
        return 1.0
    if cp_loss <= 75:
        return 0.7
    if cp_loss <= 150:
        return 0.3
    return 0.05

def _negative_weight_from_cp_loss(cp_loss):
    if cp_loss == UNKNOWN_CP_LOSS:
        return 0.0
    return min(1.0, max(0.25, float(cp_loss) / 600.0))

def _cp_loss_bucket(cp_loss):
    if cp_loss == UNKNOWN_CP_LOSS:
        return CP_LOSS_BUCKET_UNKNOWN
    if cp_loss >= CRITICAL_CP_LOSS_THRESHOLD:
        return CP_LOSS_BUCKET_CRITICAL
    if cp_loss >= BLUNDER_CP_LOSS_THRESHOLD:
        return CP_LOSS_BUCKET_BLUNDER
    if cp_loss >= MISTAKE_CP_LOSS_THRESHOLD:
        return CP_LOSS_BUCKET_MISTAKE
    if cp_loss >= INACCURACY_CP_LOSS_THRESHOLD:
        return CP_LOSS_BUCKET_INACCURACY
    return CP_LOSS_BUCKET_OK

def _is_bad_move_from_cp_loss(cp_loss):
    return cp_loss != UNKNOWN_CP_LOSS and cp_loss >= BAD_MOVE_CP_LOSS_THRESHOLD

def _piece_value_cp(piece):
    if piece is None:
        return 0
    return _PIECE_VALUES_CP.get(piece.piece_type, 0)

def _search_score_after_move_cp(board, move, search_depth):
    """Evaluate one candidate move with shallow alpha-beta from mover POV."""
    mover = board.turn
    board.push(move)
    try:
        depth_remaining = max(0, search_depth - 1)
        white_score = chess_player.alphabetahelper(
            board.turn, board, depth_remaining, -10000, 10000
        )
    finally:
        board.pop()

    mover_score = white_score if mover == chess.WHITE else -white_score
    return float(mover_score) * 100.0

def _candidate_risk_score(board, move):
    """Prefer moves likely to expose hanging-piece mistakes."""
    moving_piece = board.piece_at(move.from_square)
    score = _piece_value_cp(moving_piece)

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        score += 100 + _piece_value_cp(victim)
    if board.gives_check(move):
        score += 50
    if move.promotion:
        score += _PIECE_VALUES_CP.get(move.promotion, 0)

    board.push(move)
    try:
        moved_piece = board.piece_at(move.to_square)
        if moved_piece is not None and board.is_attacked_by(
                board.turn, move.to_square):
            score += 200 + _piece_value_cp(moved_piece)
    finally:
        board.pop()

    return score

def _select_search_negative_candidates(board, played_move, max_candidates):
    candidates = [move for move in board.legal_moves if move != played_move]
    scored = [
        (_candidate_risk_score(board, move), move.uci(), move)
        for move in candidates
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [move for _, _, move in scored[:max_candidates]]

def _search_assisted_negative_moves(board, played_move):
    if (
        SEARCH_NEGATIVE_CANDIDATES <= 0 or
        SEARCH_NEGATIVE_MAX_PER_POSITION <= 0 or
        SEARCH_NEGATIVE_DEPTH <= 0
    ):
        return []

    try:
        reference_score = _search_score_after_move_cp(
            board, played_move, SEARCH_NEGATIVE_DEPTH
        )
    except Exception:
        return []

    negatives = []
    candidates = _select_search_negative_candidates(
        board, played_move, SEARCH_NEGATIVE_CANDIDATES
    )
    for candidate in candidates:
        try:
            candidate_score = _search_score_after_move_cp(
                board, candidate, SEARCH_NEGATIVE_DEPTH
            )
        except Exception:
            continue

        search_loss = max(0.0, reference_score - candidate_score)
        if search_loss >= SEARCH_NEGATIVE_THRESHOLD:
            negatives.append((candidate, search_loss))
            if len(negatives) >= SEARCH_NEGATIVE_MAX_PER_POSITION:
                break

    return negatives

def _bucket_name(bucket):
    return {
        CP_LOSS_BUCKET_UNKNOWN: 'unknown',
        CP_LOSS_BUCKET_OK: 'ok',
        CP_LOSS_BUCKET_INACCURACY: 'inaccuracy',
        CP_LOSS_BUCKET_MISTAKE: 'mistake',
        CP_LOSS_BUCKET_BLUNDER: 'blunder',
        CP_LOSS_BUCKET_CRITICAL: 'critical',
    }.get(int(bucket), 'unknown')

def _training_policy_color(headers):
    color_name = headers.get('TrainingPolicyColor') or headers.get('StrongSide')
    if color_name is None:
        return None
    color_name = color_name.strip().lower()
    if color_name == 'white':
        return chess.WHITE
    if color_name == 'black':
        return chess.BLACK
    return None

def _should_save_positive(policy_color_mode, training_color, side_to_move):
    if policy_color_mode == 'all':
        return True
    if policy_color_mode == 'tagged':
        return training_color is not None and side_to_move == training_color
    raise ValueError(f"Unsupported policy_color_mode: {policy_color_mode!r}")

def _played_eval_cp(board, move, engine, pov_color):
    board.push(move)
    try:
        if board.is_checkmate():
            return MATE_SCORE_CP
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        info = engine.analyse(board, _engine_limit())
        return _score_to_cp(info['score'], pov_color)
    finally:
        board.pop()

def _calculate_cp_loss(board, move, comment, next_comment, engine,
                       best_eval_cp=None):
    """Return centipawn loss for the played move, from side-to-move's POV."""
    try:
        pov_color = board.turn
        if best_eval_cp is None:
            best_eval_cp = _position_eval_cp(board, comment, engine)
            if best_eval_cp is None:
                return UNKNOWN_CP_LOSS

        played_eval_cp = _parse_eval_annotation(next_comment, pov_color)
        if played_eval_cp is None:
            if engine is None:
                return UNKNOWN_CP_LOSS
            played_eval_cp = _played_eval_cp(board, move, engine, pov_color)

        return max(0.0, float(best_eval_cp - played_eval_cp))
    except Exception:
        return UNKNOWN_CP_LOSS

def open_pgn(source, pgn_name=None):
    """Open a PGN file from a .pgn path or inside a .zip. Returns a text file object."""
    if isinstance(source, str) and source.endswith('.pgn'):
        return open(source, encoding='utf-8', errors='replace')

    if isinstance(source, str) and source.endswith('.zip'):
        zf = zipfile.ZipFile(source, 'r')
        binary = zf.open(pgn_name)
        return io.TextIOWrapper(binary, encoding='utf-8', errors='replace')

    raise ValueError(f"Unsupported source type: {source!r}")

def count_games(source, pgn_name=None):
    """Count games in a PGN by counting [Event tags."""
    count = 0
    if isinstance(source, str) and source.endswith('.zip'):
        with zipfile.ZipFile(source, 'r') as zf:
            with zf.open(pgn_name) as f:
                for line in io.TextIOWrapper(f, encoding='utf-8', errors='replace'):
                    if line.startswith('[Event '):
                        count += 1
    elif isinstance(source, str) and source.endswith('.pgn'):
        with open(source, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('[Event '):
                    count += 1
    return count

def game_identity_key(headers):
    """Stable enough key to keep duplicate games in the same split."""
    return (
        headers.get('White', '').strip().lower(),
        headers.get('Black', '').strip().lower(),
        headers.get('Date', '').strip(),
        headers.get('Round', '').strip(),
        headers.get('Result', '').strip(),
    )

def collect_game_keys(source, pgn_name=None, exclude_game_keys=None):
    keys = []
    pgn_file = open_pgn(source, pgn_name)
    try:
        while True:
            headers = chess.pgn.read_headers(pgn_file)
            if headers is None:
                break
            key = game_identity_key(headers)
            if exclude_game_keys is not None and key in exclude_game_keys:
                continue
            keys.append(key)
    finally:
        pgn_file.close()
    return keys

def split_counts(total_games):
    split_total = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if not np.isclose(split_total, 1.0):
        raise ValueError(
            f"TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must be 1.0, got "
            f"{split_total:.4f}"
        )
    train_count = int(total_games * TRAIN_SPLIT)
    val_count = int(total_games * VAL_SPLIT)
    test_count = total_games - train_count - val_count
    return train_count, val_count, test_count

def _ensure_fresh_chunk_dir(path):
    existing = glob.glob(os.path.join(path, 'chunk_*.npz'))
    if existing and not ALLOW_EXISTING_CHUNKS:
        raise FileExistsError(
            f"{path!r} already contains chunk_*.npz files. Use fresh output "
            "directories or set ALLOW_EXISTING_CHUNKS=1 if you intentionally "
            "want to overwrite matching chunk names."
        )

def preprocess_pgn(source, pgn_name=None, output_dir=None,
                   sequence_length=SEQ_LEN, chunk_size=CHUNK_SIZE,
                   skip_games=0, max_games=None, chunk_offset=0,
                   calculate_cp_loss=CALCULATE_CP_LOSS,
                   exclude_game_keys=None,
                   policy_color_mode='all'):
    """Parse PGN and save played-move positions to .npz chunks.

    Returns (positions_saved, next_chunk_idx). If a node has a Stockfish
    [%best_move: ...] annotation, the saved chunk also includes whether the
    human move matched it. That metadata is not used as the training target.
    """
    os.makedirs(output_dir, exist_ok=True)

    boards_buf, moves_buf, move_idx_buf, policy_target_type_buf = [], [], [], []
    fen_buf, played_uci_buf = [], []
    legal_indices_buf = []
    played_engine_best_buf, cp_loss_buf, sample_weight_buf = [], [], []
    is_bad_move_buf, cp_loss_bucket_buf = [], []
    value_target_buf, value_source_buf = [], []
    chunk_idx              = chunk_offset
    total_pos              = 0
    game_count             = 0
    eligible_game_count    = 0
    excluded_games         = 0
    engine_best_total      = 0
    engine_best_matches    = 0
    invalid_engine_best    = 0
    illegal_played_moves   = 0
    cp_loss_total          = 0
    cp_loss_known          = 0
    value_eval_count       = 0
    value_result_count     = 0
    positive_samples       = 0
    negative_samples       = 0
    bad_move_samples       = 0
    search_negative_samples = 0
    cp_loss_bucket_counts  = {
        CP_LOSS_BUCKET_UNKNOWN: 0,
        CP_LOSS_BUCKET_OK: 0,
        CP_LOSS_BUCKET_INACCURACY: 0,
        CP_LOSS_BUCKET_MISTAKE: 0,
        CP_LOSS_BUCKET_BLUNDER: 0,
        CP_LOSS_BUCKET_CRITICAL: 0,
    }
    skipped_tagged_moves   = 0

    use_cp_loss = calculate_cp_loss and not SEARCH_NEGATIVE_ONLY
    engine = None
    if use_cp_loss:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as exc:
            print(f"  CP loss disabled: could not start Stockfish at "
                  f"{STOCKFISH_PATH!r} ({exc})")

    pgn_file = open_pgn(source, pgn_name)
    try:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_count += 1
            game_key = game_identity_key(game.headers)
            if exclude_game_keys is not None and game_key in exclude_game_keys:
                excluded_games += 1
                continue

            eligible_game_count += 1

            # Skip games before the split point
            if eligible_game_count <= skip_games:
                continue

            # Stop after max_games if set
            if (max_games is not None and
                    (eligible_game_count - skip_games) > max_games):
                break

            board = game.board()
            recent_moves = []
            training_color = _training_policy_color(game.headers)

            for node in game.mainline():
                move    = node.move
                comment = node.comment
                fen = board.fen()
                side_to_move = board.turn
                next_node = node.variations[0] if node.variations else None
                next_comment = next_node.comment if next_node is not None else ''

                best_move_match = re.search(r'\[%best_move: ([^\]]+)\]', comment)
                best_move = best_move_match.group(1) if best_move_match else None
                played_engine_best = NO_ENGINE_ANNOTATION

                if best_move is not None:
                    try:
                        engine_move = board.parse_uci(best_move)
                        engine_best_total += 1
                        if engine_move == move:
                            engine_best_matches += 1
                            played_engine_best = ENGINE_BEST_PLAYED
                        else:
                            played_engine_best = ENGINE_BEST_NOT_PLAYED
                    except ValueError:
                        invalid_engine_best += 1

                is_black = (board.turn == chess.BLACK)
                board_tensor = fen_to_tensor(fen, flip=is_black)

                try:
                    if move not in board.legal_moves:
                        raise ValueError
                    save_positive = _should_save_positive(
                        policy_color_mode, training_color, side_to_move
                    )
                    potential_negative = (
                        policy_color_mode == 'tagged' and
                        training_color is not None and
                        side_to_move != training_color and
                        INCLUDE_BAD_MOVE_NEGATIVES
                    )
                    analyze_this_move = (
                        use_cp_loss and
                        (
                            policy_color_mode == 'all' or
                            potential_negative or
                            TAGGED_POSITIVE_CP_LOSS
                        )
                    )
                    policy_idx = move_to_policy_index(move, flip=is_black)
                    legal_indices = legal_policy_indices(board, flip=is_black)
                    move_seq = move_sequence_to_vector(recent_moves, flip=is_black,
                                                       max_length=sequence_length)
                    position_eval_cp = None
                    if analyze_this_move:
                        try:
                            position_eval_cp = _position_eval_cp(board, comment, engine)
                        except Exception:
                            position_eval_cp = None
                    if position_eval_cp is None:
                        value_target = _result_value_for_color(
                            game.headers.get('Result', '*'), side_to_move
                        )
                        value_source = VALUE_FROM_RESULT
                    else:
                        value_target = _cp_to_value(position_eval_cp)
                        value_source = VALUE_FROM_EVAL
                    cp_loss = _calculate_cp_loss(board, move, comment,
                                                 next_comment, engine,
                                                 best_eval_cp=position_eval_cp) if analyze_this_move else UNKNOWN_CP_LOSS
                    target_type = POLICY_TARGET_POSITIVE
                    sample_weight = _sample_weight_from_cp_loss(cp_loss)
                    cp_bucket = _cp_loss_bucket(cp_loss)
                    is_bad_move = 1 if _is_bad_move_from_cp_loss(cp_loss) else 0
                    skip_sample = False
                    if not save_positive:
                        if (
                            potential_negative and
                            is_bad_move
                        ):
                            target_type = POLICY_TARGET_NEGATIVE
                            sample_weight = _negative_weight_from_cp_loss(cp_loss)
                        else:
                            skipped_tagged_moves += 1
                            skip_sample = True

                    save_base_sample = (
                        not skip_sample and not SEARCH_NEGATIVE_ONLY
                    )

                    if save_base_sample:
                        if cp_loss != UNKNOWN_CP_LOSS:
                            cp_loss_known += 1
                            cp_loss_total += cp_loss

                        boards_buf.append(board_tensor)
                        moves_buf.append(move_seq)
                        move_idx_buf.append(policy_idx)
                        policy_target_type_buf.append(target_type)
                        fen_buf.append(fen)
                        played_uci_buf.append(move.uci())
                        legal_indices_buf.append(legal_indices)
                        played_engine_best_buf.append(played_engine_best)
                        cp_loss_buf.append(cp_loss)
                        sample_weight_buf.append(sample_weight)
                        is_bad_move_buf.append(is_bad_move)
                        cp_loss_bucket_buf.append(cp_bucket)
                        value_target_buf.append(value_target)
                        value_source_buf.append(value_source)
                        if value_source == VALUE_FROM_EVAL:
                            value_eval_count += 1
                        else:
                            value_result_count += 1
                        if target_type == POLICY_TARGET_POSITIVE:
                            positive_samples += 1
                        else:
                            negative_samples += 1
                        if is_bad_move:
                            bad_move_samples += 1
                        cp_loss_bucket_counts[cp_bucket] = (
                            cp_loss_bucket_counts.get(cp_bucket, 0) + 1
                        )
                        total_pos += 1

                    if save_positive:
                        for search_move, search_loss in (
                                _search_assisted_negative_moves(board, move)):
                            search_policy_idx = move_to_policy_index(
                                search_move, flip=is_black
                            )
                            search_bucket = _cp_loss_bucket(search_loss)
                            search_weight = _negative_weight_from_cp_loss(
                                search_loss
                            )

                            boards_buf.append(board_tensor)
                            moves_buf.append(move_seq)
                            move_idx_buf.append(search_policy_idx)
                            policy_target_type_buf.append(
                                POLICY_TARGET_NEGATIVE
                            )
                            fen_buf.append(fen)
                            played_uci_buf.append(search_move.uci())
                            legal_indices_buf.append(legal_indices)
                            played_engine_best_buf.append(NO_ENGINE_ANNOTATION)
                            cp_loss_buf.append(search_loss)
                            sample_weight_buf.append(search_weight)
                            is_bad_move_buf.append(1)
                            cp_loss_bucket_buf.append(search_bucket)
                            value_target_buf.append(value_target)
                            value_source_buf.append(value_source)

                            if value_source == VALUE_FROM_EVAL:
                                value_eval_count += 1
                            else:
                                value_result_count += 1
                            cp_loss_known += 1
                            cp_loss_total += search_loss
                            negative_samples += 1
                            bad_move_samples += 1
                            search_negative_samples += 1
                            cp_loss_bucket_counts[search_bucket] = (
                                cp_loss_bucket_counts.get(search_bucket, 0) + 1
                            )
                            total_pos += 1
                except ValueError:
                    illegal_played_moves += 1

                recent_moves.append(move)
                if len(recent_moves) > sequence_length:
                    recent_moves.pop(0)
                    
                board.push(move)

                if len(boards_buf) >= chunk_size:
                    _save_chunk(output_dir, chunk_idx,
                                boards_buf, moves_buf, move_idx_buf,
                                policy_target_type_buf,
                                legal_indices_buf,
                                fen_buf, played_uci_buf,
                                played_engine_best_buf, cp_loss_buf,
                                sample_weight_buf,
                                is_bad_move_buf, cp_loss_bucket_buf,
                                value_target_buf, value_source_buf)
                    chunk_idx += 1
                    boards_buf, moves_buf, move_idx_buf, policy_target_type_buf = [], [], [], []
                    fen_buf, played_uci_buf = [], []
                    legal_indices_buf = []
                    played_engine_best_buf, cp_loss_buf, sample_weight_buf = [], [], []
                    is_bad_move_buf, cp_loss_bucket_buf = [], []
                    value_target_buf, value_source_buf = [], []
                    print(f"  chunk_{chunk_idx:04d}.npz  |  "
                          f"{game_count:,} games processed  |  "
                          f"{total_pos:,} positions saved",
                          flush=True)
    finally:
        pgn_file.close()
        if engine is not None:
            engine.quit()

    # Flush remaining positions
    if boards_buf:
        _save_chunk(output_dir, chunk_idx,
                    boards_buf, moves_buf, move_idx_buf,
                    policy_target_type_buf, legal_indices_buf,
                    fen_buf, played_uci_buf,
                    played_engine_best_buf, cp_loss_buf,
                    sample_weight_buf,
                    is_bad_move_buf, cp_loss_bucket_buf,
                    value_target_buf, value_source_buf)
        chunk_idx += 1

    if engine_best_total:
        match_rate = engine_best_matches / engine_best_total
        print(f"  Engine-best match rate: {engine_best_matches:,}/"
              f"{engine_best_total:,} ({match_rate:.2%})")
    if cp_loss_known:
        loss_label = (
            "Average CP/search loss"
            if search_negative_samples else
            "Average CP loss"
        )
        print(f"  {loss_label}: {cp_loss_total / cp_loss_known:.1f} "
              f"over {cp_loss_known:,} positions")
    if value_eval_count or value_result_count:
        print(f"  Value targets: {value_eval_count:,} from eval, "
              f"{value_result_count:,} from game result")
    print(f"  Policy targets: {positive_samples:,} positive, "
          f"{negative_samples:,} negative")
    if search_negative_samples and SEARCH_NEGATIVE_ONLY:
        print(f"  Bad-move labels: {bad_move_samples:,} saved from "
              "search-assisted negatives only")
    else:
        print(f"  Bad-move labels: {bad_move_samples:,} saved at "
              f">= {BAD_MOVE_CP_LOSS_THRESHOLD:.0f} CP loss")
    if search_negative_samples:
        print(f"  Search-assisted negatives: {search_negative_samples:,} "
              f"from depth-{SEARCH_NEGATIVE_DEPTH} alpha-beta")
    bucket_summary = ', '.join(
        f"{_bucket_name(bucket)}={count:,}"
        for bucket, count in sorted(cp_loss_bucket_counts.items())
        if count
    )
    if bucket_summary:
        print(f"  CP loss buckets: {bucket_summary}")
    if skipped_tagged_moves:
        print(f"  Skipped tagged non-target moves: {skipped_tagged_moves:,}")
    if invalid_engine_best:
        print(f"  Ignored invalid engine annotations: {invalid_engine_best:,}")
    if illegal_played_moves:
        print(f"  Skipped illegal PGN moves: {illegal_played_moves:,}")
    if excluded_games:
        print(f"  Skipped duplicate games already present in another source: "
              f"{excluded_games:,}")

    return total_pos, chunk_idx

def _save_chunk(output_dir, chunk_idx, boards, moves, move_idx,
                policy_target_type,
                legal_indices, fen, played_uci,
                played_engine_best, cp_loss, sample_weight,
                is_bad_move, cp_loss_bucket,
                value_target, value_source):
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    legal_lengths = np.array([len(indices) for indices in legal_indices],
                             dtype=np.int32)
    legal_offsets = np.zeros(len(legal_lengths) + 1, dtype=np.int32)
    legal_offsets[1:] = np.cumsum(legal_lengths)
    if legal_indices:
        legal_move_indices = np.concatenate(legal_indices).astype(np.int32)
    else:
        legal_move_indices = np.array([], dtype=np.int32)

    np.savez_compressed(
        path,
        boards  = np.array(boards,  dtype=np.float32),
        moves   = np.array(moves,   dtype=np.float32),
        move_idx = np.array(move_idx, dtype=np.int32),
        policy_target_type = np.array(policy_target_type, dtype=np.int8),
        fen = np.array(fen, dtype=np.str_),
        played_uci = np.array(played_uci, dtype=np.str_),
        legal_move_indices = legal_move_indices,
        legal_move_offsets = legal_offsets,
        played_engine_best = np.array(played_engine_best, dtype=np.int8),
        cp_loss = np.array(cp_loss, dtype=np.float32),
        sample_weight = np.array(sample_weight, dtype=np.float32),
        is_bad_move = np.array(is_bad_move, dtype=np.int8),
        cp_loss_bucket = np.array(cp_loss_bucket, dtype=np.int8),
        value_target = np.array(value_target, dtype=np.float32),
        value_source = np.array(value_source, dtype=np.int8),
    )

def _section(title, source, pgn_name=None):
    print("=" * 60)
    print(title)
    name = pgn_name or source
    print(f"  Source : {source}  ->  {name}")
    print("=" * 60)

def preprocess_default_sources():
    for output_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        _ensure_fresh_chunk_dir(output_dir)

    print("Counting games to calculate train/val/test splits...")
    gm_game_keys = collect_game_keys(GM_ZIP, GM_EVAL_PGN)
    gm_game_key_set = set(gm_game_keys)
    magnus_all_game_keys = collect_game_keys(MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_game_keys = [
        key for key in magnus_all_game_keys
        if key not in gm_game_key_set
    ]
    skipped_magnus_duplicates = len(magnus_all_game_keys) - len(magnus_game_keys)

    total_gm = len(gm_game_keys)
    total_magnus = len(magnus_game_keys)
    gm_train_count, gm_val_count, gm_test_count = split_counts(total_gm)
    magnus_train_count, magnus_val_count, magnus_test_count = split_counts(total_magnus)

    print(f"  Split ratios       : train={TRAIN_SPLIT:.0%}  val={VAL_SPLIT:.0%}  test={TEST_SPLIT:.0%}")
    print(f"  GM games           : {total_gm:,}  ->  "
          f"{gm_train_count:,}/{gm_val_count:,}/{gm_test_count:,}")
    print(f"  Magnus games       : {total_magnus:,}  ->  "
          f"{magnus_train_count:,}/{magnus_val_count:,}/{magnus_test_count:,}")
    if skipped_magnus_duplicates:
        print(f"  Magnus duplicates  : skipped {skipped_magnus_duplicates:,} "
              "games already found in the GM archive")
    print()

    train_chunk = val_chunk = test_chunk = 0
    train_pos_total = val_pos_total = test_pos_total = 0

    _section("Step 1/6 - GM games (training)", GM_ZIP, GM_EVAL_PGN)
    gm_train_pos, train_chunk = preprocess_pgn(
        source=GM_ZIP,
        pgn_name=GM_EVAL_PGN,
        output_dir=TRAIN_DIR,
        max_games=gm_train_count,
        chunk_offset=train_chunk,
    )
    train_pos_total += gm_train_pos
    print(f"\n  GM training: {gm_train_pos:,} positions saved.\n")

    _section("Step 2/6 - GM games (validation)", GM_ZIP, GM_EVAL_PGN)
    gm_val_pos, val_chunk = preprocess_pgn(
        source=GM_ZIP,
        pgn_name=GM_EVAL_PGN,
        output_dir=VAL_DIR,
        skip_games=gm_train_count,
        max_games=gm_val_count,
        chunk_offset=val_chunk,
    )
    val_pos_total += gm_val_pos
    print(f"\n  GM validation: {gm_val_pos:,} positions saved.\n")

    _section("Step 3/6 - GM games (test)", GM_ZIP, GM_EVAL_PGN)
    gm_test_pos, test_chunk = preprocess_pgn(
        source=GM_ZIP,
        pgn_name=GM_EVAL_PGN,
        output_dir=TEST_DIR,
        skip_games=gm_train_count + gm_val_count,
        max_games=gm_test_count,
        chunk_offset=test_chunk,
    )
    test_pos_total += gm_test_pos
    print(f"\n  GM test: {gm_test_pos:,} positions saved.\n")

    _section("Step 4/6 - Magnus games (training)", MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_train_pos, train_chunk = preprocess_pgn(
        source=MAGNUS_ZIP,
        pgn_name=MAGNUS_EVAL_PGN,
        output_dir=TRAIN_DIR,
        max_games=magnus_train_count,
        chunk_offset=train_chunk,
        exclude_game_keys=gm_game_key_set,
    )
    train_pos_total += magnus_train_pos
    print(f"\n  Magnus training: {magnus_train_pos:,} positions saved.\n")

    _section("Step 5/6 - Magnus games (validation)", MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_val_pos, val_chunk = preprocess_pgn(
        source=MAGNUS_ZIP,
        pgn_name=MAGNUS_EVAL_PGN,
        output_dir=VAL_DIR,
        skip_games=magnus_train_count,
        max_games=magnus_val_count,
        chunk_offset=val_chunk,
        exclude_game_keys=gm_game_key_set,
    )
    val_pos_total += magnus_val_pos
    print(f"\n  Magnus validation: {magnus_val_pos:,} positions saved.\n")

    _section("Step 6/6 - Magnus games (test)", MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_test_pos, test_chunk = preprocess_pgn(
        source=MAGNUS_ZIP,
        pgn_name=MAGNUS_EVAL_PGN,
        output_dir=TEST_DIR,
        skip_games=magnus_train_count + magnus_val_count,
        max_games=magnus_test_count,
        chunk_offset=test_chunk,
        exclude_game_keys=gm_game_key_set,
    )
    test_pos_total += magnus_test_pos
    print(f"\n  Magnus test: {magnus_test_pos:,} positions saved.\n")

    print("=" * 60)
    print("Preprocessing complete.")
    print(f"  Training positions   : {train_pos_total:,}")
    print(f"  Validation positions : {val_pos_total:,}")
    print(f"  Test positions       : {test_pos_total:,}")
    print("Run neural_network.py to train, then evaluate_model.py on the test set.")
    print("=" * 60)

def preprocess_single_source(args):
    _ensure_fresh_chunk_dir(args.output_dir)
    _section("Single PGN preprocessing", args.single_pgn)
    positions, next_chunk = preprocess_pgn(
        source=args.single_pgn,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        chunk_size=args.chunk_size,
        skip_games=args.skip_games,
        max_games=args.max_games,
        chunk_offset=args.chunk_offset,
        calculate_cp_loss=not args.no_cp_loss,
        policy_color_mode=args.policy_color_mode,
    )
    print("=" * 60)
    print("Single-source preprocessing complete.")
    print(f"  Positions saved : {positions:,}")
    print(f"  Next chunk idx   : {next_chunk}")
    print(f"  Output dir       : {args.output_dir}")
    print("=" * 60)

def main():
    global SEARCH_NEGATIVE_CANDIDATES
    global SEARCH_NEGATIVE_MAX_PER_POSITION
    global SEARCH_NEGATIVE_DEPTH
    global SEARCH_NEGATIVE_THRESHOLD
    global SEARCH_NEGATIVE_ONLY

    parser = argparse.ArgumentParser(
        description='Preprocess chess PGNs into neural-network chunk files.'
    )
    parser.add_argument('--single_pgn', default=None,
                        help='Preprocess one plain .pgn file instead of the default GM/Magnus archives.')
    parser.add_argument('--output_dir', default=TRAIN_DIR,
                        help='Output chunk directory for --single_pgn.')
    parser.add_argument('--policy_color_mode', choices=('all', 'tagged'),
                        default='all',
                        help='all = train every played move; tagged = train only TrainingPolicyColor/StrongSide and save bad opposite-side moves as negatives.')
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--skip_games', type=int, default=0)
    parser.add_argument('--chunk_offset', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--sequence_length', type=int, default=SEQ_LEN)
    parser.add_argument('--no_cp_loss', action='store_true',
                        help='Disable Stockfish CP-loss analysis. Tagged bad-move negatives require CP loss.')
    parser.add_argument('--search_negative_candidates', type=int,
                        default=SEARCH_NEGATIVE_CANDIDATES,
                        help='Number of non-played legal moves to shallow-search per saved positive position. 0 disables search-assisted negatives.')
    parser.add_argument('--search_negative_max_per_position', type=int,
                        default=SEARCH_NEGATIVE_MAX_PER_POSITION,
                        help='Maximum search-assisted negative targets to save for one position.')
    parser.add_argument('--search_negative_depth', type=int,
                        default=SEARCH_NEGATIVE_DEPTH,
                        help='Alpha-beta depth used to label search-assisted negative moves.')
    parser.add_argument('--search_negative_threshold', type=float,
                        default=SEARCH_NEGATIVE_THRESHOLD,
                        help='Minimum shallow-search loss, in centipawn-like units, before saving a candidate as a negative target.')
    parser.add_argument('--search_negative_only', action='store_true',
                        default=SEARCH_NEGATIVE_ONLY,
                        help='Save only search-assisted negative targets; useful for augmenting existing positive/CP-loss chunks without duplicating them.')
    args = parser.parse_args()

    SEARCH_NEGATIVE_CANDIDATES = args.search_negative_candidates
    SEARCH_NEGATIVE_MAX_PER_POSITION = args.search_negative_max_per_position
    SEARCH_NEGATIVE_DEPTH = args.search_negative_depth
    SEARCH_NEGATIVE_THRESHOLD = args.search_negative_threshold
    SEARCH_NEGATIVE_ONLY = args.search_negative_only

    if args.single_pgn:
        preprocess_single_source(args)
    else:
        preprocess_default_sources()

if __name__ == '__main__':
    main()
