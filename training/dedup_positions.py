"""Cap duplicated positions in training chunks before a retrain.

Why: in data/train_chunks, opening positions (moves 1-10) are 82% duplicates
(70% of samples are positions seen 5+ times), while positions after move 20
are ~99% unique. Training loss is therefore dominated by re-memorizing
opening theory instead of learning to generalize in the middlegame — which
is exactly where the model collapses (top-1 64% -> 37%, blunder rate
0.7% -> 14%).

This tool streams chunk files and writes new chunks where each unique
position+played-move pair appears at most --max_repeats times (different
played moves from the same position are kept as distinct entries, preserving
the empirical move distribution's support). All metadata fields are carried
over unchanged, so the output is a drop-in TRAIN_DIR.

Usage:
  python -m training.dedup_positions \
      --input_dir data/train_chunks \
      --output_dir data/train_chunks_dedup \
      --max_repeats 4
"""

import argparse
import glob
import os
from collections import Counter

import numpy as np

# 'moves' is only present in v2 chunks (v3 has no LSTM history); the writer
# uses whichever of these keys the input chunk actually has.
ALL_PER_ROW_KEYS = (
    'boards', 'moves', 'move_idx', 'policy_target_type', 'fen', 'played_uci',
    'played_engine_best', 'cp_loss', 'sample_weight', 'is_bad_move',
    'cp_loss_bucket', 'value_target', 'value_source',
)


def position_key(fen: str, played_uci: str) -> str:
    # Position sans halfmove/fullmove clocks + the move that was played.
    return ' '.join(fen.split(' ')[:4]) + '|' + played_uci


def _write_chunk(output_dir, chunk_idx, rows, legal_rows):
    arrays = {}
    first = rows[0]
    row_keys = [k for k in ALL_PER_ROW_KEYS if k in first]
    for key in row_keys:
        values = [r[key] for r in rows]
        if key in ('fen', 'played_uci'):
            arrays[key] = np.array(values, dtype=np.str_)
        else:
            arrays[key] = np.stack([np.asarray(v) for v in values]) \
                if np.asarray(first[key]).ndim else np.array(values)
    lengths = np.array([len(li) for li in legal_rows], dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    arrays['legal_move_indices'] = (
        np.concatenate(legal_rows).astype(np.int32)
        if legal_rows else np.array([], dtype=np.int32)
    )
    arrays['legal_move_offsets'] = offsets
    arrays['board_encoding'] = rows[0]['board_encoding']
    path = os.path.join(output_dir, f'chunk_{chunk_idx:04d}.npz')
    np.savez_compressed(path, **arrays)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/train_chunks')
    parser.add_argument('--output_dir', default='data/train_chunks_dedup')
    parser.add_argument('--max_repeats', type=int, default=4,
                        help='Keep at most N copies of each position+move pair.')
    parser.add_argument('--chunk_size', type=int, default=50_000)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, 'chunk_*.npz')))
    if not paths:
        raise SystemExit(f'No chunks in {args.input_dir!r}')
    os.makedirs(args.output_dir, exist_ok=True)
    if glob.glob(os.path.join(args.output_dir, 'chunk_*.npz')):
        raise SystemExit(f'{args.output_dir!r} already contains chunks; '
                         'use a fresh directory.')

    seen = Counter()
    kept = dropped = 0
    out_rows, out_legal = [], []
    chunk_idx = 0

    for path in paths:
        with np.load(path) as data:
            present_keys = [k for k in ALL_PER_ROW_KEYS if k in data]
            arrays = {key: data[key] for key in present_keys}
            legal_indices = data['legal_move_indices']
            legal_offsets = data['legal_move_offsets']
            encoding = data['board_encoding']

        n = len(arrays['fen'])
        for i in range(n):
            key = position_key(str(arrays['fen'][i]),
                               str(arrays['played_uci'][i]))
            seen[key] += 1
            if seen[key] > args.max_repeats:
                dropped += 1
                continue
            row = {k: arrays[k][i] for k in present_keys}
            row['board_encoding'] = encoding
            out_rows.append(row)
            out_legal.append(
                legal_indices[legal_offsets[i]:legal_offsets[i + 1]]
            )
            kept += 1
            if len(out_rows) >= args.chunk_size:
                out_path = _write_chunk(args.output_dir, chunk_idx,
                                        out_rows, out_legal)
                print(f'  wrote {out_path} (kept {kept:,} / '
                      f'dropped {dropped:,})', flush=True)
                chunk_idx += 1
                out_rows, out_legal = [], []
        print(f'{os.path.basename(path)} done | kept {kept:,} '
              f'dropped {dropped:,}', flush=True)

    if out_rows:
        _write_chunk(args.output_dir, chunk_idx, out_rows, out_legal)
        chunk_idx += 1

    total = kept + dropped
    print(f'\nDone. {total:,} rows in -> {kept:,} kept '
          f'({dropped:,} dropped, {dropped / max(total, 1):.1%}) '
          f'across {chunk_idx} chunks in {args.output_dir}')
    print('Use it for the next retrain with: '
          f'TRAIN_DIR={args.output_dir}')


if __name__ == '__main__':
    main()
