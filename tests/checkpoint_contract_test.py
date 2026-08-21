"""Fast regression tests for the serving checkpoint contract."""

import os
import sys
import tempfile
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_model import _validate_hf_revision, load_trained_model
from neural_network import (
    BOARD_ENCODING_VERSION,
    BOARD_ENCODING_VERSION_V3,
    ChessModelV3,
    ChessModelV4,
)


def _payload(model):
    return {
        'checkpoint_schema_version': 1,
        'model_state_dict': model.state_dict(),
        'board_encoding': BOARD_ENCODING_VERSION_V3,
        'arch_version': 'v4',
        'residual_filters': model.filters,
        'residual_blocks': model.blocks,
    }


def test_architecture_encoding_mismatch_fails_closed():
    model = ChessModelV4(filters=8, blocks=1)
    payload = _payload(model)
    payload['board_encoding'] = BOARD_ENCODING_VERSION
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'bad-encoding.pt')
        torch.save(payload, path)
        try:
            load_trained_model(path)
        except ValueError as exc:
            assert 'requires board encoding' in str(exc)
        else:
            raise AssertionError('architecture/encoding mismatch was accepted')


def test_checksum_mismatch_fails_closed():
    model = ChessModelV4(filters=8, blocks=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.pt')
        torch.save(_payload(model), path)
        with mock.patch.dict(
            os.environ,
            {'MAGNUS_MODEL_SHA256': '0' * 64},
            clear=False,
        ):
            try:
                load_trained_model(path)
            except RuntimeError as exc:
                assert 'SHA-256 mismatch' in str(exc)
            else:
                raise AssertionError('wrong checkpoint digest was accepted')


def test_unexpected_state_key_fails_closed():
    model = ChessModelV4(filters=8, blocks=1)
    payload = _payload(model)
    payload['model_state_dict']['not_a_real_layer'] = torch.zeros(1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'unexpected.pt')
        torch.save(payload, path)
        try:
            load_trained_model(path)
        except RuntimeError as exc:
            assert 'unexpected tensors' in str(exc)
        else:
            raise AssertionError('unexpected state tensor was accepted')


def test_dimensionless_legacy_checkpoint_uses_env_knob_defaults():
    # Legacy dimensions come from training-time environment settings.
    model = ChessModelV3(filters=16, blocks=2)
    payload = {
        'model_state_dict': model.state_dict(),
        'board_encoding': BOARD_ENCODING_VERSION_V3,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'legacy-v3.pt')
        torch.save(payload, path)
        with mock.patch.multiple(
            'load_model', RESIDUAL_FILTERS=16, RESIDUAL_BLOCKS=2
        ):
            loaded = load_trained_model(path)
    assert loaded.filters == 16 and loaded.blocks == 2


def test_hf_download_requires_immutable_revision():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, 'missing.pt')
        env = {
            'MAGNUS_HF_REPO': 'example/model',
            'MAGNUS_ALLOW_FLOATING_HF_REVISION': '0',
        }
        with mock.patch.dict(os.environ, env, clear=True):
            try:
                load_trained_model(missing)
            except ValueError as exc:
                assert 'MAGNUS_HF_REVISION' in str(exc)
            else:
                raise AssertionError('floating Hugging Face revision was accepted')


def test_hf_revision_rejects_branch_and_tag_names():
    assert _validate_hf_revision('A' * 40) == 'a' * 40
    assert _validate_hf_revision('b' * 64) == 'b' * 64
    for mutable in ('main', 'latest', 'v1.0', 'abc123'):
        try:
            _validate_hf_revision(mutable)
        except ValueError:
            pass
        else:
            raise AssertionError(f'mutable revision {mutable!r} was accepted')


if __name__ == '__main__':
    test_architecture_encoding_mismatch_fails_closed()
    test_checksum_mismatch_fails_closed()
    test_unexpected_state_key_fails_closed()
    test_dimensionless_legacy_checkpoint_uses_env_knob_defaults()
    test_hf_download_requires_immutable_revision()
    test_hf_revision_rejects_branch_and_tag_names()
    print('checkpoint contract tests passed')
