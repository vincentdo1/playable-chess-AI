"""Audit test 7: evaluation/serving really loads the intended checkpoint.

For every checkpoint in model/:
  - a STRICT state-dict load into the architecture selected by the stored
    board_encoding must succeed (exact key and shape match with today's code â€”
    no silently skipped or randomly initialized layers);
  - stored residual_filters/residual_blocks must match the architecture the
    loaders would build right now;
  - load_trained_model() must hand back weights BIT-IDENTICAL to the file's
    tensors, with the right encoding spec attached and deterministic
    eval-mode outputs.

Then regression-tests three fixed hazards: load_trained_model rejects a
checkpoint missing whole layers rather than serving a partly random network;
the default MODEL_PATH points at v3; and evaluate_model.py follows the loaded
checkpoint's encoding.
"""

import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import (
    BOARD_ENCODING_VERSION, BOARD_ENCODING_VERSION_V3, ChessModel,
    ChessModelV3, ChessModelV4, ChunkDataset, RESIDUAL_BLOCKS,
    RESIDUAL_FILTERS, get_encoding_spec,
)
from load_model import MODEL_PATH, load_trained_model

SCRATCH = os.environ.get('AUDIT_SCRATCH', tempfile.gettempdir())


def _build_for_checkpoint(ckpt):
    """Mirror load_model's dispatch: arch_version first, then encoding."""
    if ckpt.get('arch_version') == 'v4':
        return ChessModelV4(
            filters=ckpt.get('residual_filters', 256),
            blocks=ckpt.get('residual_blocks', 12),
        )
    encoding = ckpt.get('board_encoding') or BOARD_ENCODING_VERSION
    if encoding == BOARD_ENCODING_VERSION_V3:
        return ChessModelV3()
    if encoding == BOARD_ENCODING_VERSION:
        return ChessModel()
    raise AssertionError(f'unknown encoding {encoding!r}')


def test_all_checkpoints_load_strictly():
    paths = sorted(glob.glob('model/*.pt'))
    if not paths:
        try:
            import pytest
            pytest.skip('artifact integration test requires model/*.pt')
        except ImportError:
            print('  SKIP strict checkpoint audit: no model/*.pt artifacts')
            return
    for path in paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        encoding = ckpt.get('board_encoding') or BOARD_ENCODING_VERSION
        model = _build_for_checkpoint(ckpt)

        # Strict load: raises on ANY missing/unexpected key or shape mismatch.
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

        if ckpt.get('arch_version') != 'v4':
            # v2/v3 build from module globals; v4 builds from its own
            # checkpoint metadata, so the global check does not apply.
            assert ckpt.get('residual_filters', RESIDUAL_FILTERS) == RESIDUAL_FILTERS, (
                f'{path}: trained with residual_filters='
                f'{ckpt.get("residual_filters")} but current code builds '
                f'{RESIDUAL_FILTERS}'
            )
            assert ckpt.get('residual_blocks', RESIDUAL_BLOCKS) == RESIDUAL_BLOCKS, (
                f'{path}: trained with residual_blocks='
                f'{ckpt.get("residual_blocks")} but current code builds '
                f'{RESIDUAL_BLOCKS}'
            )

        # The public loader must hand back bit-identical weights.
        loaded = load_trained_model(path)
        loaded_sd = loaded.state_dict()
        for name, tensor in ckpt['model_state_dict'].items():
            assert torch.equal(loaded_sd[name].cpu(), tensor.cpu()), (
                f'{path}: parameter {name} differs after load_trained_model'
            )
        assert loaded.encoding_version == encoding
        assert loaded.encoding_spec == get_encoding_spec(encoding)
        assert loaded.value_head_trained is True

        # Deterministic eval-mode forward (dropout must be off).
        spec = get_encoding_spec(encoding)
        probe = torch.randn(2, spec['board_channels'], 8, 8)
        history = torch.zeros(
            (2, 10, 132) if spec['uses_move_history'] else (2, 0)
        )
        device = next(loaded.parameters()).device
        probe, history = probe.to(device), history.to(device)
        with torch.no_grad():
            p1, v1 = loaded(probe, history)
            p2, v2 = loaded(probe, history)
        assert torch.equal(p1, p2) and torch.equal(v1, v2), (
            f'{path}: eval-mode forward is not deterministic'
        )
        assert p1.shape == (2, spec['move_vocab_size'])
        print(f'  {path}: strict load OK, encoding={encoding}, '
              f'weights bit-identical, deterministic eval forward')


def test_partial_checkpoint_rejected():
    """Regression: a checkpoint missing whole layers must refuse to load."""
    src = 'model/grandmaster_resnet_v3.pt'
    if not os.path.exists(src):
        print('  SKIP partial-load test: v3 checkpoint missing')
        return
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    full_sd = ckpt['model_state_dict']
    doctored = {k: v for k, v in full_sd.items()
                if not k.startswith('policy_head')}
    removed = len(full_sd) - len(doctored)
    ckpt['model_state_dict'] = doctored
    tmp = os.path.join(SCRATCH, '_audit_partial_ckpt.pt')
    torch.save(ckpt, tmp)
    try:
        try:
            load_trained_model(tmp)
        except RuntimeError as exc:
            assert ('does not cover the current architecture' in str(exc) or
                    'partially initialized' in str(exc))
            print(f'  partial checkpoint (missing {removed} policy-head '
                  f'tensors) correctly rejected: {str(exc)[:60]}...')
        else:
            raise AssertionError(
                'load_trained_model accepted a checkpoint with missing '
                'layers — the strict-load hardening regressed'
            )
    finally:
        os.remove(tmp)


def test_default_model_path_is_v3():
    if os.environ.get('MODEL_PATH'):
        print(f'  SKIP default-path test: MODEL_PATH env set '
              f'({os.environ["MODEL_PATH"]!r})')
        return
    assert os.path.basename(MODEL_PATH) == 'grandmaster_resnet_v3.pt', (
        f'default MODEL_PATH regressed to {MODEL_PATH!r}; backend and every '
        'load_trained_model() call without a path would serve the wrong model'
    )
    print(f'  default MODEL_PATH serves the v3 checkpoint: {MODEL_PATH}')


def test_evaluate_model_wiring_measures_v3():
    """Regression: held-out eval follows the checkpoint's encoding."""
    if not os.path.exists('model/grandmaster_resnet_v3.pt'):
        print('  SKIP evaluate_model wiring test: v3 checkpoint missing')
        return
    from evaluation.evaluate_model import make_collate_policy_batch  # re-export check
    model = load_trained_model('model/grandmaster_resnet_v3.pt')
    encoding = model.encoding_version
    spec = model.encoding_spec
    ds = ChunkDataset('data/test_chunks_v3', shuffle=False,
                      expected_encoding=encoding)
    ds.chunk_paths = ds.chunk_paths[:1]
    collate = make_collate_policy_batch(spec['move_vocab_size'])
    it = iter(ds)
    batch = collate([next(it) for _ in range(8)])
    boards, moves, targets, legal_mask, *_ = batch
    device = next(model.parameters()).device
    with torch.no_grad():
        policy, value = model(boards.to(device), moves.to(device))
    assert policy.shape == (8, spec['move_vocab_size'])
    print('  evaluate_model wiring: v3 model + v3 chunks + v3 vocab forward OK')


if __name__ == '__main__':
    test_all_checkpoints_load_strictly()
    test_partial_checkpoint_rejected()
    test_default_model_path_is_v3()
    test_evaluate_model_wiring_measures_v3()
    print('audit checkpoint loading tests passed')
