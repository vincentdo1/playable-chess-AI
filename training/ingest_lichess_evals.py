"""Build v4 distillation shards from the Lichess position-evaluations dump.

Source: https://huggingface.co/datasets/Lichess/chess-position-evaluations
(mirror of https://database.lichess.org/ evaluations; CC0). Raw schema per row:
fen, line (PV, UCI), depth, knodes, cp, mate.

Sign convention (verified empirically on shard 0, 2026-07-02 — see
tests/distill_ingest_test.py): **cp and mate are from White's point of view.**
This tool converts them to the side-to-move convention used everywhere in this
repo (value_target = tanh(cp_stm / 600), mate -> +/-1) so downstream code never
sees the White-POV numbers.

Output: compact parquet shards with columns (fen, move, value_target, depth) —
board tensors are intentionally NOT materialized; the training DataLoader
encodes FENs on the fly with the audited v3 codecs (docs/ROADMAP_2500.md WS1).
Deduplication keeps the deepest evaluation per unique FEN. Validation rows come
exclusively from hash-bucket 0, so train/val never share a position.

Usage:
  python -m training.ingest_lichess_evals --num_shards 3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import zlib
from datetime import datetime

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.preprocess import VALUE_CP_SCALE  # 600, shared with v2/v3 labels

# King-takes-rook castling notation used by Lichess PVs; anything else in the
# move column is already standard UCI. (Rook e1->h1 moves also match these
# strings, which is why normalization goes through parse_uci per row.)
_CASTLE_UCI_CANDIDATES = frozenset(('e1h1', 'e1a1', 'e8h8', 'e8a8'))

HF_SHARD_URL = (
    'https://huggingface.co/datasets/Lichess/chess-position-evaluations/'
    'resolve/main/data/data_{index:04d}.parquet'
)

OUT_SCHEMA = pa.schema([
    ('fen', pa.string()),
    ('move', pa.string()),
    ('value_target', pa.float32()),
    ('depth', pa.uint8()),
])


def _download_shard(raw_dir: str, index: int) -> str:
    path = os.path.join(raw_dir, f'data_{index:04d}.parquet')
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f'  raw shard {index} already present', flush=True)
        return path
    url = HF_SHARD_URL.format(index=index)
    print(f'  downloading shard {index}: {url}', flush=True)
    subprocess.run(
        ['curl', '-sL', '--fail', '-o', path, url], check=True
    )
    return path


def _flush_bucket(tmp_dir: str, bucket: int, part: int, rows: list) -> None:
    fens, moves, values, depths = zip(*rows)
    table = pa.Table.from_arrays(
        [
            pa.array(fens, pa.string()),
            pa.array(moves, pa.string()),
            pa.array(values, pa.float32()),
            pa.array(depths, pa.uint8()),
        ],
        schema=OUT_SCHEMA,
    )
    path = os.path.join(tmp_dir, f'bucket{bucket:02d}_part{part:04d}.parquet')
    pq.write_table(table, path, compression='zstd')


def partition_shard(raw_path: str, tmp_dir: str, num_buckets: int,
                    min_depth: int, part_counters: list,
                    stats: dict, flush_rows: int = 750_000) -> None:
    """Stream one raw shard into per-bucket part files, converting labels."""
    buffers: list[list] = [[] for _ in range(num_buckets)]
    pf = pq.ParquetFile(raw_path)
    for batch in pf.iter_batches(
        batch_size=131_072, columns=['fen', 'line', 'depth', 'cp', 'mate']
    ):
        fens = batch.column('fen').to_pylist()
        lines = batch.column('line').to_pylist()
        depths = batch.column('depth').to_pylist()
        cps = batch.column('cp').to_pylist()
        mates = batch.column('mate').to_pylist()
        stats['rows_in'] += len(fens)
        for fen, line, depth, cp, mate in zip(fens, lines, depths, cps, mates):
            if depth is None or depth < min_depth:
                stats['dropped_depth'] += 1
                continue
            if not line:
                stats['dropped_no_line'] += 1
                continue
            fields = fen.split(' ')
            if len(fields) < 2:
                stats['dropped_bad_fen'] += 1
                continue
            black_to_move = fields[1] == 'b'
            if mate is not None:
                white_value = 1.0 if mate > 0 else -1.0
                stats['mate_rows'] += 1
            elif cp is not None:
                white_value = float(np.tanh(cp / VALUE_CP_SCALE))
            else:
                stats['dropped_no_eval'] += 1
                continue
            value = -white_value if black_to_move else white_value
            move = line.split(' ', 1)[0]
            if move in _CASTLE_UCI_CANDIDATES:
                # Lichess PVs encode castling as king-takes-rook (e1h1);
                # normalize to standard UCI so consumers can stay strict.
                # Board construction only for these ~1.5% of rows.
                try:
                    move = chess.Board(fen).parse_uci(move).uci()
                except (ValueError, KeyError):
                    stats['dropped_bad_castle'] += 1
                    continue
            bucket = zlib.crc32(fen.encode()) % num_buckets
            buffers[bucket].append((fen, move, value, depth))
            if len(buffers[bucket]) >= flush_rows:
                _flush_bucket(tmp_dir, bucket, part_counters[bucket],
                              buffers[bucket])
                part_counters[bucket] += 1
                buffers[bucket] = []
    for bucket, rows in enumerate(buffers):
        if rows:
            _flush_bucket(tmp_dir, bucket, part_counters[bucket], rows)
            part_counters[bucket] += 1


class ShardWriter:
    """Accumulates rows and writes fixed-size output shards."""

    def __init__(self, out_dir: str, prefix: str, rows_per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.rows: list = []
        self.index = 0
        self.total = 0

    def add_many(self, rows: list) -> None:
        self.rows.extend(rows)
        while len(self.rows) >= self.rows_per_shard:
            self._write(self.rows[:self.rows_per_shard])
            self.rows = self.rows[self.rows_per_shard:]

    def close(self) -> None:
        if self.rows:
            self._write(self.rows)
            self.rows = []

    def _write(self, rows: list) -> None:
        fens, moves, values, depths = zip(*rows)
        table = pa.Table.from_arrays(
            [
                pa.array(fens, pa.string()),
                pa.array(moves, pa.string()),
                pa.array(values, pa.float32()),
                pa.array(depths, pa.uint8()),
            ],
            schema=OUT_SCHEMA,
        )
        path = os.path.join(
            self.out_dir, f'{self.prefix}_{self.index:04d}.parquet'
        )
        pq.write_table(table, path, compression='zstd')
        self.index += 1
        self.total += len(rows)
        print(f'  wrote {path} ({len(rows):,} rows)', flush=True)


def dedupe_and_emit(tmp_dir: str, out_dir: str, num_buckets: int,
                    val_positions: int, rows_per_shard: int,
                    rng: np.random.Generator, stats: dict) -> None:
    train_writer = ShardWriter(out_dir, 'train', rows_per_shard)
    val_writer = ShardWriter(out_dir, 'val', rows_per_shard)
    for bucket in range(num_buckets):
        best: dict = {}
        part_paths = sorted(glob.glob(
            os.path.join(tmp_dir, f'bucket{bucket:02d}_part*.parquet')))
        for path in part_paths:
            table = pq.read_table(path)
            for fen, move, value, depth in zip(
                table.column('fen').to_pylist(),
                table.column('move').to_pylist(),
                table.column('value_target').to_pylist(),
                table.column('depth').to_pylist(),
            ):
                prev = best.get(fen)
                if prev is None or depth > prev[2]:
                    best[fen] = (move, value, depth)
        # Free the bucket's temp space immediately: at full-dump scale the
        # temp partition is ~14 GB, and reclaiming it bucket-by-bucket keeps
        # peak disk usage roughly flat while the output grows.
        for path in part_paths:
            os.remove(path)
        rows = [(fen, m, v, d) for fen, (m, v, d) in best.items()]
        stats['unique_positions'] += len(rows)
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        if bucket == 0 and val_positions > 0:
            val_writer.add_many(rows[:val_positions])
            rows = rows[val_positions:]
        train_writer.add_many(rows)
        print(f'  bucket {bucket:02d}: {len(best):,} unique positions',
              flush=True)
    train_writer.close()
    val_writer.close()
    stats['train_rows'] = train_writer.total
    stats['val_rows'] = val_writer.total


def main():
    parser = argparse.ArgumentParser(
        description='Lichess evaluations -> v4 distillation shards.'
    )
    parser.add_argument('--num_shards', type=int, default=3,
                        help='How many ~2.1GB raw shards to ingest.')
    parser.add_argument('--first_shard', type=int, default=0)
    parser.add_argument('--raw_dir', default='data/lichess_evals_raw')
    parser.add_argument('--out_dir', default='data/distill_chunks_v4')
    parser.add_argument('--min_depth', type=int, default=12)
    parser.add_argument('--val_positions', type=int, default=250_000)
    parser.add_argument('--buckets', type=int, default=64,
                        help='Hash-partition count for dedupe. Sized so one '
                             'bucket dict fits in RAM: the full 20-shard dump '
                             '(~388M uniques) needs >=64 on a 16 GB machine.')
    parser.add_argument('--rows_per_shard', type=int, default=2_000_000)
    parser.add_argument('--keep_raw', action='store_true',
                        help='Keep raw shards after processing (default: '
                             'delete to save disk).')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if glob.glob(os.path.join(args.out_dir, '*.parquet')):
        raise SystemExit(f'{args.out_dir!r} already contains shards; '
                         'use a fresh directory.')
    tmp_dir = os.path.join(args.out_dir, '_tmp_buckets')
    os.makedirs(tmp_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    stats = {
        'rows_in': 0, 'dropped_depth': 0, 'dropped_no_line': 0,
        'dropped_bad_fen': 0, 'dropped_no_eval': 0, 'dropped_bad_castle': 0,
        'mate_rows': 0, 'unique_positions': 0,
    }
    part_counters = [0] * args.buckets

    start = datetime.now()
    for i in range(args.first_shard, args.first_shard + args.num_shards):
        raw_path = _download_shard(args.raw_dir, i)
        print(f'  partitioning shard {i}...', flush=True)
        partition_shard(raw_path, tmp_dir, args.buckets, args.min_depth,
                        part_counters, stats)
        if not args.keep_raw:
            os.remove(raw_path)
        print(f'  shard {i} done | rows_in={stats["rows_in"]:,}', flush=True)

    print('deduplicating buckets and writing output shards...', flush=True)
    dedupe_and_emit(tmp_dir, args.out_dir, args.buckets, args.val_positions,
                    args.rows_per_shard, rng, stats)
    shutil.rmtree(tmp_dir)

    stats['min_depth'] = args.min_depth
    stats['value_convention'] = (
        'side-to-move POV, tanh(cp/600), mate=+/-1; source cp was White-POV '
        'and negated for black-to-move (verified vs mate-in-1 probes)'
    )
    stats['elapsed_seconds'] = (datetime.now() - start).total_seconds()
    with open(os.path.join(args.out_dir, 'ingest_stats.json'), 'w',
              encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
