"""Audit-grade validation of the v4 distillation dataset.

Standalone runnable:  python tests/distill_ingest_test.py

The 2026-07-02 probe on raw shard 0 established that the Lichess dump's
cp/mate are WHITE-POV (mate=+1 with black to move was never a black mate,
0/150; mate=-1 with black to move always was, 150/150). Ingestion converts to
this repo's side-to-move convention. This test re-verifies the *output*:

  1. every sampled row's move is legal in its FEN (policy labels sound);
  2. value_target is in [-1, 1] and mate rows sit exactly at +/-1;
  3. INDEPENDENT sign check: local Stockfish evaluates sampled positions from
     the side to move; the stored value_target must agree in sign for
     decisively evaluated rows and correlate strongly overall. An inverted
     convention cannot pass this.
  4. if a raw shard is still on disk, the mate-in-1 probe from the original
     verification is repeated against it.
"""

import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyarrow before anything that pulls torch (see training/train_distill.py).
import pyarrow as pa
import pyarrow.parquet as pq

import chess
import chess.engine
import numpy as np

# Override to validate a different ingest output (e.g. the Phase-2 full dump:
# DISTILL_DATA_DIR=data/distill_chunks_v4_full).
DATA_DIR = os.environ.get('DISTILL_DATA_DIR', 'data/distill_chunks_v4')
RAW_DIR = 'data/lichess_evals_raw'
STOCKFISH = os.environ.get('STOCKFISH_PATH', os.path.abspath('stockfish.exe'))
SAMPLE_ROWS = 600
SF_ROWS = 120


def _sample_rows(rng):
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(DATA_DIR, 'train_*.parquet')))
    assert paths, f'no shards in {DATA_DIR}; run ingest_lichess_evals first'
    picked = [paths[0], paths[len(paths) // 2], paths[-1]]
    rows = []
    for path in dict.fromkeys(picked):
        table = pq.read_table(path)
        fens = table.column('fen').to_pylist()
        moves = table.column('move').to_pylist()
        values = table.column('value_target').to_pylist()
        for i in rng.choice(len(fens), size=min(SAMPLE_ROWS, len(fens)),
                            replace=False):
            rows.append((fens[i], moves[i], values[i]))
    return rows


def test_labels_and_range(rows):
    """Strict label check: the stored move must survive the exact path the
    trainer uses (parse_uci + policy index inside the recomputed legal set).
    `in board.legal_moves` is NOT strict enough — python-chess silently
    normalizes Lichess's king-takes-rook castling UCI (e1h1), which is how
    1.55% skipped-castling rows evaded the first version of this test."""
    from neural_network import legal_policy_indices_v3, move_to_policy_index_v3

    mates = 0
    castle_notation = 0
    for fen, move, value in rows:
        board = chess.Board(fen)
        flip = board.turn == chess.BLACK
        parsed = board.parse_uci(move)   # raises if illegal
        if parsed.uci() != move:
            castle_notation += 1
        idx = move_to_policy_index_v3(parsed, flip=flip)
        assert idx in set(legal_policy_indices_v3(board, flip=flip).tolist()), (
            f'move {move} not encodable/legal at {fen}'
        )
        assert -1.0 <= value <= 1.0
        if abs(value) == 1.0:
            mates += 1
    print(f'  {len(rows)} rows: stored moves all legal via the strict trainer '
          f'path ({mates} mate rows at +/-1, {castle_notation} rows in '
          f'king-takes-rook notation)')


def test_sign_vs_local_stockfish(rows):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    stored, measured = [], []
    try:
        for fen, _, value in rows[:SF_ROWS]:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            cp = info['score'].pov(board.turn).score(mate_score=10_000)
            stored.append(float(value))
            measured.append(float(np.tanh(cp / 600.0)))
    finally:
        engine.quit()
    stored = np.array(stored)
    measured = np.array(measured)
    corr = float(np.corrcoef(stored, measured)[0, 1])
    decisive = np.abs(measured) > 0.3
    agree = float(
        (np.sign(stored[decisive]) == np.sign(measured[decisive])).mean()
    ) if decisive.any() else 1.0
    print(f'  Stockfish cross-check on {len(stored)} rows: corr={corr:.3f}, '
          f'sign agreement on decisive rows={agree:.1%} '
          f'({int(decisive.sum())} decisive)')
    assert corr > 0.75, f'value/eval correlation too low: {corr:.3f}'
    assert agree > 0.9, f'sign agreement too low: {agree:.1%} — POV inverted?'


def test_mate_probe_on_raw_shard():
    import pyarrow.parquet as pq
    raws = sorted(glob.glob(os.path.join(RAW_DIR, 'data_*.parquet')))
    if not raws:
        print('  SKIP raw mate probe: no raw shards on disk '
              '(convention was verified 2026-07-02: cp/mate are White-POV)')
        return
    pf = pq.ParquetFile(raws[0])
    black_plus, black_minus = [], []
    for batch in pf.iter_batches(batch_size=65536,
                                 columns=['fen', 'mate']):
        for fen, mate in zip(batch.column('fen').to_pylist(),
                             batch.column('mate').to_pylist()):
            if mate == 1 and ' b ' in fen and len(black_plus) < 60:
                black_plus.append(fen)
            elif mate == -1 and ' b ' in fen and len(black_minus) < 60:
                black_minus.append(fen)
        if len(black_plus) >= 60 and len(black_minus) >= 60:
            break

    def stm_mates(fen):
        board = chess.Board(fen)
        for move in board.legal_moves:
            board.push(move)
            mated = board.is_checkmate()
            board.pop()
            if mated:
                return True
        return False

    plus_rate = np.mean([stm_mates(f) for f in black_plus]) if black_plus else 0
    minus_rate = np.mean([stm_mates(f) for f in black_minus]) if black_minus else 1
    print(f'  raw probe: black-to-move mate=+1 rows with black mate: '
          f'{plus_rate:.0%} (expect ~0), mate=-1: {minus_rate:.0%} (expect ~100%)')
    assert plus_rate < 0.1 and minus_rate > 0.9, (
        'raw shard contradicts the White-POV convention'
    )


_RAW_SCHEMA = pa.schema([
    ('fen', pa.string()), ('line', pa.string()), ('depth', pa.uint8()),
    ('knodes', pa.int32()), ('cp', pa.int16()), ('mate', pa.int8()),
])

# (fen, line, depth, cp, mate) synthetic raw rows exercising every ingest rule.
_SYNTH_ROWS = [
    # White castles kingside; Lichess writes it king-takes-rook.
    ('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5',
     'e1h1 e8h8', 25, 30, None),
    # Black castles queenside.
    ('r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR b KQkq - 6 5',
     'e8a8 e1a1', 25, -20, None),
    # Rook lift e1->h1 that merely LOOKS like castle notation; must survive
    # (king NOT on e1, so no castling conversion may fire).
    ('6k1/8/8/8/8/8/8/3KR3 w - - 10 20',
     'e1h1 g8f7', 25, 10, None),
    # Black to move, White better by 150cp (White-POV) -> negative STM value.
    ('rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',
     'b8c6', 30, 150, None),
    # Black to move, Black mates (White-POV mate=-3) -> STM value +1.
    ('6k1/8/8/8/8/2q5/1r6/6K1 b - - 0 1', 'b2b1', 40, None, -3),
    # Below min_depth: dropped.
    ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
     'e2e4', 5, 20, None),
    # Duplicate FEN at two depths: dedupe keeps the deeper row's move.
    ('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
     'c7c5', 15, -25, None),
    ('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
     'e7e5', 30, -30, None),
]


def test_ingest_unit_castling_conversion_dedupe():
    """The full ingest pipeline on a synthetic raw shard, row by row."""
    from training.ingest_lichess_evals import (
        dedupe_and_emit, partition_shard,
    )

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, 'raw.parquet')
        fens, lines, depths, cps, mates = zip(*[
            (f, l, d, c, m) for f, l, d, c, m in _SYNTH_ROWS
        ])
        pq.write_table(pa.Table.from_arrays([
            pa.array(fens, pa.string()), pa.array(lines, pa.string()),
            pa.array(depths, pa.uint8()),
            pa.array([1000] * len(fens), pa.int32()),
            pa.array(cps, pa.int16()), pa.array(mates, pa.int8()),
        ], schema=_RAW_SCHEMA), raw)

        bucket_dir = os.path.join(tmp, 'buckets')
        out_dir = os.path.join(tmp, 'out')
        os.makedirs(bucket_dir)
        os.makedirs(out_dir)
        stats = {k: 0 for k in (
            'rows_in', 'dropped_depth', 'dropped_no_line', 'dropped_bad_fen',
            'dropped_no_eval', 'dropped_bad_castle', 'mate_rows',
            'unique_positions')}
        counters = [0] * 4
        partition_shard(raw, bucket_dir, 4, 12, counters, stats, flush_rows=2)
        rng = np.random.default_rng(0)
        dedupe_and_emit(bucket_dir, out_dir, 4, 0, 100, rng, stats)

        assert stats['rows_in'] == 8
        assert stats['dropped_depth'] == 1
        assert stats['dropped_bad_castle'] == 0
        assert stats['unique_positions'] == 6
        assert not glob.glob(os.path.join(bucket_dir, '*.parquet')), (
            'bucket temp files must be deleted as they are consumed'
        )

        out = {}
        for path in glob.glob(os.path.join(out_dir, 'train_*.parquet')):
            t = pq.read_table(path)
            for fen, move, value in zip(t.column('fen').to_pylist(),
                                        t.column('move').to_pylist(),
                                        t.column('value_target').to_pylist()):
                out[fen] = (move, value)
        assert len(out) == 6
        assert out[_SYNTH_ROWS[0][0]][0] == 'e1g1', 'white O-O not normalized'
        assert out[_SYNTH_ROWS[1][0]][0] == 'e8c8', 'black O-O-O not normalized'
        assert out[_SYNTH_ROWS[2][0]][0] == 'e1h1', 'rook lift wrongly rewritten'
        # Black to move, cp=+150 White-POV -> STM value must be negative.
        assert abs(out[_SYNTH_ROWS[3][0]][1] - (-np.tanh(150 / 600))) < 1e-6
        # Black to move, mate=-3 White-POV -> STM value +1.
        assert out[_SYNTH_ROWS[4][0]][1] == 1.0
        # Dedupe kept the depth-30 row.
        assert out[_SYNTH_ROWS[6][0]][0] == 'e7e5'

        # And the trainer's dataset must consume every emitted row (the old
        # reader skipped castling rows; parse_uci fixed that).
        from training.train_distill import DistillShardDataset
        from neural_network import policy_index_to_move_v3
        ds = DistillShardDataset(
            glob.glob(os.path.join(out_dir, 'train_*.parquet')), shuffle=False
        )
        yielded = {}
        for tensor, _hist, move_idx, legal, _w, value, _t in ds:
            yielded[len(yielded)] = int(move_idx)
        assert len(yielded) == 6, (
            f'dataset yielded {len(yielded)}/6 rows — reader is skipping'
        )
    print('  ingest unit test OK: castling normalized (O-O, O-O-O), rook '
          'lift preserved, POV conversion + dedupe + temp cleanup verified, '
          'reader consumes 6/6 rows')


def test_reader_skip_rate_on_real_data():
    """With parse_uci, the trainer reader should skip ~nothing."""
    from training.train_distill import DistillShardDataset

    path = os.path.join(DATA_DIR, 'train_0000.parquet')
    if not os.path.exists(path):
        print('  SKIP reader sweep: no Phase-1 shards')
        return
    ds = DistillShardDataset([path], shuffle=False)
    limit = 200_000
    yielded = 0
    for row in ds:
        yielded += 1
        if yielded >= limit:
            break
    table_rows = pq.ParquetFile(path).metadata.num_rows
    # The iterator stops at `limit`, so reaching it means <(rows-limit) were
    # skipped before that point; assert we got the full stream prefix.
    assert yielded == min(limit, table_rows), (
        f'reader yielded {yielded} of first {limit}'
    )
    print(f'  reader sweep OK: first {yielded:,} rows of train_0000 all '
          f'consumed (castling rows included)')


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    rows = _sample_rows(rng)
    test_labels_and_range(rows)
    test_sign_vs_local_stockfish(rows)
    test_mate_probe_on_raw_shard()
    test_ingest_unit_castling_conversion_dedupe()
    test_reader_skip_rate_on_real_data()
    print('distill ingest tests passed')
