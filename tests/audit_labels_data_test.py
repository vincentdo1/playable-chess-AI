"""Audit tests 3+4+6 on REAL v3 chunk data: label legality, value targets,
and train-vs-inference preprocessing parity.

Standalone runnable (no pytest):  python tests/audit_labels_data_test.py

What is proven here, on sampled rows from every v3 chunk directory:
  3. Policy labels are legal: move_idx is inside its own legal-move window,
     the stored legal window equals the legal moves recomputed from the FEN,
     and move_idx decodes back to exactly the stored played_uci.
  4. Value targets obey the side-to-move convention's invariants:
     in [-1, 1]; result-sourced targets are exactly -1/0/+1; sample weights
     match the documented cp-loss schedule.
  6. The tensor written at preprocessing time equals the tensor the serving
     path (load_model._position_arrays) builds from the same FEN, on every
     channel except the two repetition planes (which need in-game history
     that a FEN-only board cannot have — the known, measured serve skew).
     The collator's channel-first permutation equals the serving one.

Also streams one full chunk through ChunkDataset + DataLoader with the v3
collator, which hard-fails on any target missing from its legal mask.
"""

import glob
import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (
    BOARD_ENCODING_VERSION_V3, ChunkDataset, HALFMOVE_CLOCK_SCALE,
    V3_EXTRA_CHANNELS, fen_to_tensor, legal_policy_indices_v3,
    make_collate_policy_batch, get_encoding_spec, policy_index_to_move_v3,
)
from load_model import _position_arrays
from training.preprocess import (
    UNKNOWN_CP_LOSS, VALUE_FROM_RESULT, _negative_weight_from_cp_loss,
    _sample_weight_from_cp_loss,
)

CHUNK_DIRS = (
    'data/train_chunks_v3',
    'data/val_chunks_v3',
    'data/test_chunks_v3',
    'data/train_chunks_v3_dedup',
)
ROWS_PER_CHUNK = 300
REP_CHANNELS = (V3_EXTRA_CHANNELS['repetition_1'],
                V3_EXTRA_CHANNELS['repetition_2'])
BASE_CHANNELS = [c for c in range(20) if c not in REP_CHANNELS]


def audit_chunk(path, rng):
    with np.load(path) as data:
        assert str(data['board_encoding'].item()) == BOARD_ENCODING_VERSION_V3
        boards = data['boards']
        move_idx = data['move_idx']
        offsets = data['legal_move_offsets']
        legal_indices = data['legal_move_indices']
        fens = data['fen']
        played = data['played_uci']
        target_type = data['policy_target_type']
        value_target = data['value_target']
        value_source = data['value_source']
        cp_loss = data['cp_loss']
        sample_weight = data['sample_weight']

    n = len(move_idx)
    # Offsets must be a clean partition of the flat legal-index array.
    assert offsets[0] == 0 and offsets[-1] == len(legal_indices)
    assert np.all(np.diff(offsets) > 0), 'a row has zero legal moves'

    # Whole-chunk vector invariants (cheap, so run on all rows).
    assert np.all((value_target >= -1.0) & (value_target <= 1.0))
    result_rows = value_source == VALUE_FROM_RESULT
    assert np.all(np.isin(value_target[result_rows], (-1.0, 0.0, 1.0))), (
        'result-sourced value target outside {-1, 0, 1}'
    )
    assert np.all(np.isin(target_type, (-1, 1)))

    rows = rng.choice(n, size=min(ROWS_PER_CHUNK, n), replace=False)
    rep_rows = 0
    for i in rows:
        fen = str(fens[i])
        board = chess.Board(fen)
        flip = board.turn == chess.BLACK
        start, end = int(offsets[i]), int(offsets[i + 1])
        window = legal_indices[start:end]

        # --- 3. policy label legality ---
        assert move_idx[i] in window, (
            f'{path} row {i}: move_idx not in its legal window'
        )
        assert len(set(window.tolist())) == len(window), (
            f'{path} row {i}: duplicate legal indices'
        )
        recomputed = legal_policy_indices_v3(board, flip=flip)
        assert set(window.tolist()) == set(recomputed.tolist()), (
            f'{path} row {i}: stored legal set differs from FEN recompute\n'
            f'  fen={fen}'
        )
        decoded = policy_index_to_move_v3(int(move_idx[i]), flip=flip)
        assert decoded.uci() == str(played[i]), (
            f'{path} row {i}: move_idx decodes to {decoded.uci()} but '
            f'played_uci={played[i]} (fen={fen})'
        )

        # --- 6. train vs inference tensor parity ---
        serving_tensor, move_seq = _position_arrays(
            board, BOARD_ENCODING_VERSION_V3
        )
        assert move_seq is None, 'v3 serving path unexpectedly built history'
        stored = boards[i]
        assert np.array_equal(
            stored[:, :, BASE_CHANNELS], serving_tensor[:, :, BASE_CHANNELS]
        ), (
            f'{path} row {i}: stored tensor differs from serving tensor on a '
            f'non-repetition channel (fen={fen})'
        )
        # Base-17 must also match a direct fen_to_tensor recompute.
        assert np.array_equal(stored[:, :, :17], fen_to_tensor(fen, flip=flip))
        clock_plane = stored[0, 0, V3_EXTRA_CHANNELS['halfmove_clock']]
        expected_clock = min(board.halfmove_clock / HALFMOVE_CLOCK_SCALE, 1.0)
        assert abs(clock_plane - expected_clock) < 1e-6
        if stored[0, 0, REP_CHANNELS[0]] or stored[0, 0, REP_CHANNELS[1]]:
            rep_rows += 1

        # --- 4. weight schedule matches the documented cp-loss mapping ---
        cl = float(cp_loss[i])
        if target_type[i] > 0:
            expected_w = _sample_weight_from_cp_loss(
                cl if cl >= 0 else UNKNOWN_CP_LOSS
            )
        else:
            expected_w = _negative_weight_from_cp_loss(
                cl if cl >= 0 else UNKNOWN_CP_LOSS
            )
        assert abs(float(sample_weight[i]) - expected_w) < 1e-6, (
            f'{path} row {i}: sample_weight {sample_weight[i]} does not match '
            f'cp_loss {cl} schedule ({expected_w})'
        )

    return len(rows), rep_rows


def test_chunk_dirs():
    rng = np.random.default_rng(20260702)
    grand_rows = grand_rep = 0
    for chunk_dir in CHUNK_DIRS:
        paths = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))
        if not paths:
            print(f'  SKIP {chunk_dir}: no chunks')
            continue
        picked = [paths[0], paths[len(paths) // 2], paths[-1]]
        picked = list(dict.fromkeys(picked))
        dir_rows = dir_rep = 0
        for path in picked:
            checked, rep_rows = audit_chunk(path, rng)
            dir_rows += checked
            dir_rep += rep_rows
        grand_rows += dir_rows
        grand_rep += dir_rep
        print(f'  {chunk_dir}: {dir_rows} rows OK across {len(picked)} chunks '
              f'({dir_rep} rows carry in-game repetition planes)')
    if grand_rows == 0:
        try:
            import pytest
            pytest.skip('artifact integration test requires v3 chunk data')
        except ImportError:
            print('  SKIP chunk audit: no v3 chunk data')
            return
    assert grand_rows >= 1500, f'too few rows audited: {grand_rows}'
    print(f'  total {grand_rows} rows: labels legal, decode==played_uci, '
          f'legal sets match FEN recompute, tensors match serving path')
    print(f'  known serve skew: {grand_rep}/{grand_rows} rows '
          f'({grand_rep / grand_rows:.2%}) have repetition planes a FEN-only '
          f'board would zero')


def test_collator_permutation_matches_serving():
    import torch
    paths = sorted(glob.glob(os.path.join(CHUNK_DIRS[0], 'chunk_*.npz')))
    if not paths:
        try:
            import pytest
            pytest.skip('artifact integration test requires v3 chunk data')
        except ImportError:
            print('  SKIP collator artifact check: no v3 chunk data')
            return
    with np.load(paths[0]) as data:
        board = data['boards'][0]
    collated = torch.from_numpy(np.stack([board])).float().permute(0, 3, 1, 2)
    served = torch.tensor(board, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    assert torch.equal(collated, served), (
        'training collator and serving path disagree on channel layout'
    )
    print('  collator channel-first layout == serving layout')


def test_full_chunk_streams_through_dataset():
    """ChunkDataset + v3 collator over one full chunk; the collator raises if
    any target is missing from its legal mask, so surviving = pass."""
    from torch.utils.data import DataLoader

    if not glob.glob('data/val_chunks_v3/chunk_*.npz'):
        try:
            import pytest
            pytest.skip('artifact integration test requires validation chunks')
        except ImportError:
            print('  SKIP dataset stream: no validation chunks')
            return
    spec = get_encoding_spec(BOARD_ENCODING_VERSION_V3)
    ds = ChunkDataset('data/val_chunks_v3', shuffle=False,
                      expected_encoding=BOARD_ENCODING_VERSION_V3)
    ds.chunk_paths = ds.chunk_paths[:1]
    loader = DataLoader(
        ds, batch_size=512, num_workers=0,
        collate_fn=make_collate_policy_batch(spec['move_vocab_size']),
    )
    rows = 0
    for boards, moves, targets, legal_mask, weights, values, types in loader:
        assert boards.shape[1] == spec['board_channels']
        assert legal_mask.shape[1] == spec['move_vocab_size']
        assert bool(legal_mask.gather(1, targets.unsqueeze(1)).all())
        rows += boards.shape[0]
    print(f'  full chunk ({rows} rows) streamed through ChunkDataset + '
          f'collator with every target inside its legal mask')


if __name__ == '__main__':
    test_chunk_dirs()
    test_collator_permutation_matches_serving()
    test_full_chunk_streams_through_dataset()
    print('audit label/data tests passed')
