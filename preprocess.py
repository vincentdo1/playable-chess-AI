"""Converts PGN files to .npz chunks for fast training. Run once before training."""

import io
import os
import re
import zipfile
import numpy as np
import chess
import chess.pgn

from neural_network import fen_to_tensor, move_to_index, move_sequence_to_vector, flip_square

# Config — edit paths to match your setup
GM_ZIP      = 'C:/Users/Vincent/Downloads/GM_games_2600.zip'
MAGNUS_ZIP  = 'C:/Users/Vincent/Downloads/magnus.zip'

GM_EVAL_PGN     = 'GM_games_eval.pgn'      # name of the eval file inside GM_ZIP
MAGNUS_EVAL_PGN = 'magnus_eval.pgn'        # name of the eval file inside MAGNUS_ZIP

TRAIN_DIR  = 'data/train_chunks'
VAL_DIR    = 'data/val_chunks'

CHUNK_SIZE       = 50_000  # positions per .npz file
SEQ_LEN          = 10
MAGNUS_VAL_SPLIT = 0.20    # last 20% of Magnus games go to validation

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

def preprocess_pgn(source, pgn_name=None, output_dir=None,
                   sequence_length=SEQ_LEN, chunk_size=CHUNK_SIZE,
                   skip_games=0, max_games=None, chunk_offset=0):
    """Parse PGN and save positions to .npz chunks. Returns (positions_saved, next_chunk_idx)."""
    os.makedirs(output_dir, exist_ok=True)

    boards_buf, moves_buf, from_buf, to_buf = [], [], [], []
    chunk_idx  = chunk_offset
    total_pos  = 0
    game_count = 0

    pgn_file = open_pgn(source, pgn_name)
    try:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_count += 1

            # Skip games before the split point
            if game_count <= skip_games:
                continue

            # Stop after max_games if set
            if max_games is not None and (game_count - skip_games) > max_games:
                break

            board = game.board()
            recent_moves = []

            for node in game.mainline():
                move    = node.move
                comment = node.comment

                best_move_match = re.search(r'\[%best_move: ([^\]]+)\]', comment)
                best_move = best_move_match.group(1) if best_move_match else None

                is_black = (board.turn == chess.BLACK)
                board_tensor = fen_to_tensor(board.fen(), flip=is_black)

                if best_move is not None:
                    try:
                        from_idx, to_idx = move_to_index(board.parse_uci(best_move), board)
                        if is_black:
                            from_idx = flip_square(from_idx)
                            to_idx   = flip_square(to_idx)
                        move_seq = move_sequence_to_vector(recent_moves, flip=is_black,
                                                           max_length=sequence_length)
                        boards_buf.append(board_tensor)
                        moves_buf.append(move_seq)
                        from_buf.append(from_idx)
                        to_buf.append(to_idx)
                        total_pos += 1
                    except ValueError:
                        pass  # skip rare illegal annotations

                recent_moves.append(move)
                if len(recent_moves) > sequence_length:
                    recent_moves.pop(0)
                    
                board.push(move)

                if len(boards_buf) >= chunk_size:
                    _save_chunk(output_dir, chunk_idx,
                                boards_buf, moves_buf, from_buf, to_buf)
                    chunk_idx += 1
                    boards_buf, moves_buf, from_buf, to_buf = [], [], [], []
                    print(f"  chunk_{chunk_idx:04d}.npz  |  "
                          f"{game_count:,} games processed  |  "
                          f"{total_pos:,} positions saved",
                          flush=True)
    finally:
        pgn_file.close()

    # Flush remaining positions
    if boards_buf:
        _save_chunk(output_dir, chunk_idx,
                    boards_buf, moves_buf, from_buf, to_buf)
        chunk_idx += 1

    return total_pos, chunk_idx

def _save_chunk(output_dir, chunk_idx, boards, moves, from_sq, to_sq):
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards  = np.array(boards,  dtype=np.float32),
        moves   = np.array(moves,   dtype=np.float32),
        from_sq = np.array(from_sq, dtype=np.int32),
        to_sq   = np.array(to_sq,   dtype=np.int32),
    )

def _section(title, source, pgn_name=None):
    print("=" * 60)
    print(title)
    name = pgn_name or source
    print(f"  Source : {source}  →  {name}")
    print("=" * 60)

if __name__ == '__main__':

    # Step 1: Count Magnus games to find the 80/20 split point
    print("Counting Magnus games to calculate train/val split...")
    total_magnus = count_games(MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_train_count = int(total_magnus * (1 - MAGNUS_VAL_SPLIT))
    magnus_val_start   = magnus_train_count  # games before this → train
    print(f"  Total Magnus games : {total_magnus:,}")
    print(f"  Training games     : {magnus_train_count:,}  (first {int((1-MAGNUS_VAL_SPLIT)*100)}%)")
    print(f"  Validation games   : {total_magnus - magnus_train_count:,}  (last {int(MAGNUS_VAL_SPLIT*100)}%)")
    print()

    # Step 2: Preprocess GM games → training chunks
    _section("Step 1/3 — GM games (training)", GM_ZIP, GM_EVAL_PGN)
    gm_pos, next_chunk = preprocess_pgn(
        source      = GM_ZIP,
        pgn_name    = GM_EVAL_PGN,
        output_dir  = TRAIN_DIR,
    )
    print(f"\n  GM training: {gm_pos:,} positions saved.\n")

    # Step 3: Preprocess first 80% of Magnus games → training chunks
    #         (appended after GM chunks, chunk numbering continues)
    _section("Step 2/3 — Magnus games, train split (first 80%)",
             MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_train_pos, next_chunk = preprocess_pgn(
        source       = MAGNUS_ZIP,
        pgn_name     = MAGNUS_EVAL_PGN,
        output_dir   = TRAIN_DIR,
        max_games    = magnus_train_count,
        chunk_offset = next_chunk,           # continue numbering from GM chunks
    )
    print(f"\n  Magnus training: {magnus_train_pos:,} positions saved.")
    print(f"  Total training : {gm_pos + magnus_train_pos:,} positions.\n")

    # Step 4: Preprocess last 20% of Magnus games → validation chunks
    _section("Step 3/3 — Magnus games, val split (last 20%)",
             MAGNUS_ZIP, MAGNUS_EVAL_PGN)
    magnus_val_pos, _ = preprocess_pgn(
        source      = MAGNUS_ZIP,
        pgn_name    = MAGNUS_EVAL_PGN,
        output_dir  = VAL_DIR,
        skip_games  = magnus_val_start,      # skip the training portion
    )
    print(f"\n  Magnus validation: {magnus_val_pos:,} positions saved.\n")

    print("=" * 60)
    print("Preprocessing complete. Run neural_network.py to train.")
    print("=" * 60)