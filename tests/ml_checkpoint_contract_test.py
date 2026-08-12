"""Focused ML data/checkpoint contract regressions.

Standalone runnable:  python tests/ml_checkpoint_contract_test.py

These tests intentionally exercise the production checkpoint writers and
resume helpers. Synthetic hand-written model payloads would not catch metadata
drift between a writer and load_model's architecture dispatch.
"""

import json
import os
import sys
import tempfile

# Keep pyarrow before torch for the Windows DLL ordering constraint documented
# in training/train_distill.py.
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neural_network as N
from evaluation.evaluate_model import checkpoint_objective
from experiments.train_self_play import (
    TrainingConfig,
    _gate_match,
    _restore_optimizer_checkpoint,
    _save_checkpoint as save_selfplay_checkpoint,
    _snapshot_state,
)
from load_model import load_trained_model
from training.ingest_lichess_evals import (
    OUT_SCHEMA,
    _atomic_write_json,
    _sha256_file as ingest_sha256,
    _validate_raw_shard,
    _validate_source_revision,
)
from training.train_distill import (
    _load_data_provenance,
    _restore_checkpoint as restore_distill_checkpoint,
    _save as save_distill_checkpoint,
)


def _take_optimizer_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model.cnn[0].weight.float().square().mean()
    loss.backward()
    optimizer.step()


def test_selfplay_production_checkpoint_roundtrip_and_resume():
    model = N.ChessModelV4(filters=16, blocks=1).to(N.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    _take_optimizer_step(model, optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    config = TrainingConfig(
        learning_rate=7e-5,
        value_loss_weight=1.0,
        gate_games=0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'selfplay_iter0003.pt')
        save_selfplay_checkpoint(
            model,
            optimizer,
            scaler,
            path,
            iteration=3,
            training_config=config,
            extra={'value_loss_weight': 1.0, 'iteration_seed': 13},
        )
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['arch_version'] == 'v4'
        assert payload['residual_filters'] == 16
        assert payload['residual_blocks'] == 1
        assert payload['optimizer_class'] == 'Adam'
        assert payload['training_kind'] == 'self_play_alphazero'

        loaded = load_trained_model(path)
        assert isinstance(loaded, N.ChessModelV4)
        assert loaded.filters == 16 and loaded.blocks == 1
        resumed_optimizer = torch.optim.Adam(loaded.parameters(), lr=1e-4)
        resumed_scaler = torch.amp.GradScaler('cuda', enabled=False)
        restored = _restore_optimizer_checkpoint(
            path, resumed_optimizer, resumed_scaler, config
        )
        assert restored['iteration'] == 3
        assert resumed_optimizer.state_dict()['state']
        assert resumed_optimizer.param_groups[0]['lr'] == 7e-5
    print('  self-play production writer -> public loader -> optimizer resume OK')


def test_gate_clone_preserves_custom_architecture():
    model = N.ChessModelV4(filters=16, blocks=1).to(N.DEVICE)
    config = TrainingConfig(gate_games=0)
    score = _gate_match(model, _snapshot_state(model), config)
    assert score == 0.0
    print('  self-play gate preserves custom v4 width/depth')


def test_distill_production_checkpoint_resume_contract():
    model = N.ChessModelV4(filters=16, blocks=1).to(N.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _take_optimizer_step(model, optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    provenance = {
        'kind': 'ingest_manifest',
        'path': '/corpus/ingest_manifest.json',
        'sha256': 'a' * 64,
        'source_revision': 'b' * 40,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'distill.pt')
        save_distill_checkpoint(model, optimizer, scaler, path, {
            'epoch': 2,
            'step': 123,
            'epoch_complete': False,
            'val_loss': 1.25,
            'seed': 9,
            'data_provenance': provenance,
            'training_config': {'lr': 3e-4},
        })
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['checkpoint_format_version'] == 2
        assert payload['arch_version'] == 'v4'
        assert payload['residual_filters'] == 16
        assert payload['epoch_complete'] is False
        assert payload['rng_state']['torch_cpu'].dtype == torch.uint8

        restored_model = N.ChessModelV4(filters=16, blocks=1).to(N.DEVICE)
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=3e-4
        )
        restored_scaler = torch.amp.GradScaler('cuda', enabled=False)
        restored = restore_distill_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            restored_scaler,
            provenance,
            expected_value_loss_weight=float(
                os.environ.get('VALUE_LOSS_WEIGHT', '1.0')
            ),
            expected_label_smoothing=float(N.LABEL_SMOOTHING),
        )
        assert restored['step'] == 123
        assert restored_optimizer.state_dict()['state']

        wrong_provenance = {**provenance, 'sha256': 'c' * 64}
        try:
            restore_distill_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                restored_scaler,
                wrong_provenance,
                expected_value_loss_weight=float(
                    os.environ.get('VALUE_LOSS_WEIGHT', '1.0')
                ),
                expected_label_smoothing=float(N.LABEL_SMOOTHING),
            )
        except ValueError as exc:
            assert 'current corpus' in str(exc)
        else:
            raise AssertionError('resume accepted a different data manifest')
    print('  distill safe checkpoint roundtrip + provenance guard OK')


def test_ingest_and_evaluation_metadata_contracts():
    assert _validate_source_revision('A' * 40) == 'a' * 40
    for floating in (None, '', 'main', 'latest'):
        try:
            _validate_source_revision(floating)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted floating revision {floating!r}')

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, 'raw')
        os.makedirs(raw_dir)
        raw = os.path.join(raw_dir, 'raw.parquet')
        raw_table = pa.table({
            'fen': pa.array(['8/8/8/8/8/8/8/K6k w - -']),
            'line': pa.array(['a1a2']),
            'depth': pa.array([12], type=pa.uint8()),
            'knodes': pa.array([1], type=pa.int32()),
            'cp': pa.array([0], type=pa.int16()),
            'mate': pa.array([None], type=pa.int8()),
        })
        pq.write_table(raw_table, raw)
        raw_meta = _validate_raw_shard(raw)
        assert raw_meta['rows'] == 1 and len(raw_meta['sha256']) == 64

        train = os.path.join(tmp, 'train_0000.parquet')
        val = os.path.join(tmp, 'val_0000.parquet')
        output_table = pa.Table.from_arrays([
            pa.array(['8/8/8/8/8/8/8/K6k w - -']),
            pa.array(['a1a2']),
            pa.array([0.0], type=pa.float32()),
            pa.array([12], type=pa.uint8()),
        ], schema=OUT_SCHEMA)
        pq.write_table(output_table, train)
        pq.write_table(output_table, val)
        manifest_path = os.path.join(tmp, 'ingest_manifest.json')
        _atomic_write_json(manifest_path, {
            'manifest_version': 2,
            'status': 'complete',
            'source': {'revision': 'd' * 40},
            'outputs': [
                {
                    'name': 'train_0000.parquet',
                    'rows': 1,
                    'sha256': ingest_sha256(train),
                },
                {
                    'name': 'val_0000.parquet',
                    'rows': 1,
                    'sha256': ingest_sha256(val),
                },
            ],
        })
        provenance = _load_data_provenance(tmp)
        assert provenance['kind'] == 'ingest_manifest'
        assert provenance['source_revision'] == 'd' * 40

        objective_path = os.path.join(tmp, 'objective.pt')
        torch.save({
            'value_loss_weight': 1.75,
            'label_smoothing': 0.03,
        }, objective_path)
        objective = checkpoint_objective(objective_path)
        assert objective == {
            'value_loss_weight': 1.75,
            'label_smoothing': 0.03,
            'metadata_complete': True,
        }
    print('  immutable ingest revision/manifest + eval objective contract OK')


if __name__ == '__main__':
    test_selfplay_production_checkpoint_roundtrip_and_resume()
    test_gate_clone_preserves_custom_architecture()
    test_distill_production_checkpoint_resume_contract()
    test_ingest_and_evaluation_metadata_contracts()
    print('ML checkpoint contract tests passed')
