"""Converts PGN files to .npz chunks for fast training. Run once before training.

Training labels come from the move actually played in the PGN. Optional
Stockfish annotations are kept only for analysis, such as measuring how often
the played move matched the engine's top move.
"""

import io
import glob
import os
import re
import zipfile
import numpy as np
import chess
import chess.engine
import chess.pgn

from neural_network import (
    fen_to_tensor, legal_policy_indices, move_to_index, move_sequence_to_vector,
    move_to_policy_index, flip_square
)

# Config — edit paths to match your setup
GM_ZIP = os.environ.get('GM_ZIP', 'C:/Users/Vincent/Downloads/GM_games_2600.zip')
MAGNUS_ZIP = os.environ.get('MAGNUS_ZIP', 'C:/Users/Vincent/Downloads/magnus.zip')

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

NO_ENGINE_ANNOTATION = -1
ENGINE_BEST_NOT_PLAYED = 0
ENGINE_BEST_PLAYED = 1
UNKNOWN_CP_LOSS = -1.0

def _engine_limit():
    if CP_LOSS_DEPTH is not None:
        return chess.engine.Limit(depth=CP_LOSS_DEPTH)
    return chess.engine.Limit(time=CP_LOSS_TIME_LIMIT)

def _score_to_cp(score, pov_color):
    return score.pov(pov_color).score(mate_score=MATE_SCORE_CP)

def _parse_eval_annotation(comment, pov_color):
    """Parse this project's [%eval: ...] annotation into centipawns."""
    eval_match = re.search(r'\[%eval: ([^\]]+)\]', comment)
    if not eval_match:
        return None

    value = eval_match.group(1).strip()
    try:
        if value.startswith('#'):
            mate = int(value[1:])
            white_cp = MATE_SCORE_CP if mate > 0 else -MATE_SCORE_CP
        else:
            white_cp = int(value)
    except ValueError:
        return None

    return white_cp if pov_color == chess.WHITE else -white_cp

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

def _calculate_cp_loss(board, move, comment, next_comment, engine):
    """Return centipawn loss for the played move, from side-to-move's POV."""
    try:
        pov_color = board.turn
        best_eval_cp = _parse_eval_annotation(comment, pov_color)
        if best_eval_cp is None:
            if engine is None:
                return UNKNOWN_CP_LOSS
            info = engine.analyse(board, _engine_limit())
            best_eval_cp = _score_to_cp(info['score'], pov_color)

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
                   exclude_game_keys=None):
    """Parse PGN and save played-move positions to .npz chunks.

    Returns (positions_saved, next_chunk_idx). If a node has a Stockfish
    [%best_move: ...] annotation, the saved chunk also includes whether the
    human move matched it. That metadata is not used as the training target.
    """
    os.makedirs(output_dir, exist_ok=True)

    boards_buf, moves_buf, from_buf, to_buf, move_idx_buf = [], [], [], [], []
    fen_buf, played_uci_buf = [], []
    legal_indices_buf = []
    played_engine_best_buf, cp_loss_buf, sample_weight_buf = [], [], []
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

    engine = None
    if calculate_cp_loss:
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

            for node in game.mainline():
                move    = node.move
                comment = node.comment
                fen = board.fen()
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
                    from_idx, to_idx = move_to_index(move, board)
                    policy_idx = move_to_policy_index(move, flip=is_black)
                    legal_indices = legal_policy_indices(board, flip=is_black)
                    if is_black:
                        from_idx = flip_square(from_idx)
                        to_idx   = flip_square(to_idx)
                    move_seq = move_sequence_to_vector(recent_moves, flip=is_black,
                                                       max_length=sequence_length)
                    cp_loss = _calculate_cp_loss(board, move, comment,
                                                 next_comment, engine)
                    sample_weight = _sample_weight_from_cp_loss(cp_loss)
                    if cp_loss != UNKNOWN_CP_LOSS:
                        cp_loss_known += 1
                        cp_loss_total += cp_loss

                    boards_buf.append(board_tensor)
                    moves_buf.append(move_seq)
                    from_buf.append(from_idx)
                    to_buf.append(to_idx)
                    move_idx_buf.append(policy_idx)
                    fen_buf.append(fen)
                    played_uci_buf.append(move.uci())
                    legal_indices_buf.append(legal_indices)
                    played_engine_best_buf.append(played_engine_best)
                    cp_loss_buf.append(cp_loss)
                    sample_weight_buf.append(sample_weight)
                    total_pos += 1
                except ValueError:
                    illegal_played_moves += 1

                recent_moves.append(move)
                if len(recent_moves) > sequence_length:
                    recent_moves.pop(0)
                    
                board.push(move)

                if len(boards_buf) >= chunk_size:
                    _save_chunk(output_dir, chunk_idx,
                                boards_buf, moves_buf, from_buf, to_buf,
                                move_idx_buf, legal_indices_buf,
                                fen_buf, played_uci_buf,
                                played_engine_best_buf, cp_loss_buf,
                                sample_weight_buf)
                    chunk_idx += 1
                    boards_buf, moves_buf, from_buf, to_buf, move_idx_buf = [], [], [], [], []
                    fen_buf, played_uci_buf = [], []
                    legal_indices_buf = []
                    played_engine_best_buf, cp_loss_buf, sample_weight_buf = [], [], []
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
                    boards_buf, moves_buf, from_buf, to_buf,
                    move_idx_buf, legal_indices_buf,
                    fen_buf, played_uci_buf,
                    played_engine_best_buf, cp_loss_buf,
                    sample_weight_buf)
        chunk_idx += 1

    if engine_best_total:
        match_rate = engine_best_matches / engine_best_total
        print(f"  Engine-best match rate: {engine_best_matches:,}/"
              f"{engine_best_total:,} ({match_rate:.2%})")
    if cp_loss_known:
        print(f"  Average CP loss: {cp_loss_total / cp_loss_known:.1f} "
              f"over {cp_loss_known:,} positions")
    if invalid_engine_best:
        print(f"  Ignored invalid engine annotations: {invalid_engine_best:,}")
    if illegal_played_moves:
        print(f"  Skipped illegal PGN moves: {illegal_played_moves:,}")
    if excluded_games:
        print(f"  Skipped duplicate games already present in another source: "
              f"{excluded_games:,}")

    return total_pos, chunk_idx

def _save_chunk(output_dir, chunk_idx, boards, moves, from_sq, to_sq,
                move_idx, legal_indices, fen, played_uci,
                played_engine_best, cp_loss, sample_weight):
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
        from_sq = np.array(from_sq, dtype=np.int32),
        to_sq   = np.array(to_sq,   dtype=np.int32),
        move_idx = np.array(move_idx, dtype=np.int32),
        fen = np.array(fen, dtype=np.str_),
        played_uci = np.array(played_uci, dtype=np.str_),
        legal_move_indices = legal_move_indices,
        legal_move_offsets = legal_offsets,
        played_engine_best = np.array(played_engine_best, dtype=np.int8),
        cp_loss = np.array(cp_loss, dtype=np.float32),
        sample_weight = np.array(sample_weight, dtype=np.float32),
    )

def _section(title, source, pgn_name=None):
    print("=" * 60)
    print(title)
    name = pgn_name or source
    print(f"  Source : {source}  ->  {name}")
    print("=" * 60)

def main():
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

if __name__ == '__main__':
    main()
