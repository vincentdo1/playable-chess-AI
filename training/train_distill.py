"""Train ChessModelV4 on Lichess evaluation shards.

Four-field source FENs cannot supervise clock or repetition planes, so v4
masks those inputs. Checkpoints retain the ``perspective_v3`` board encoding.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

# Import pyarrow before torch to avoid a Windows DLL crash in worker processes.
import pyarrow as pa
import pyarrow.parquet as pq

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# Configure dense engine-evaluation labels before importing neural_network.
os.environ.setdefault('VALUE_LOSS_WEIGHT', '1.0')

import neural_network as N
from neural_network import (
    BOARD_ENCODING_VERSION_V3, ChessModelV4, _NO_MOVE_HISTORY,
    board_to_tensor_v3, legal_policy_indices_v3, make_collate_policy_batch,
    mask_illegal_logits, move_to_policy_index_v3, policy_loss_for_targets,
)

MOVE_VOCAB = N.MOVE_VOCAB_SIZE_V3
CHECKPOINT_FORMAT_VERSION = 2
INGEST_MANIFEST = 'ingest_manifest.json'


class DistillShardDataset(IterableDataset):
    """Stream shards and encode FENs in DataLoader workers."""

    def __init__(self, shard_paths, shuffle=True, seed=0):
        self.shard_paths = sorted(shard_paths)
        if not self.shard_paths:
            raise FileNotFoundError('no distillation shards found')
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        paths = self.shard_paths
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        if self.shuffle:
            paths = [paths[i] for i in rng.permutation(len(paths))]
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]

        skipped = 0
        for path in paths:
            table = pq.read_table(
                path, columns=['fen', 'move', 'value_target']
            )
            fens = table.column('fen').to_pylist()
            moves = table.column('move').to_pylist()
            values = table.column('value_target').to_pylist()
            order = (rng.permutation(len(fens)) if self.shuffle
                     else range(len(fens)))
            for i in order:
                try:
                    board = chess.Board(fens[i])
                    flip = board.turn == chess.BLACK
                    # parse_uci normalizes Lichess king-takes-rook castling.
                    move = board.parse_uci(moves[i])
                    move_idx = move_to_policy_index_v3(move, flip=flip)
                    legal = legal_policy_indices_v3(board, flip=flip)
                    if move_idx not in legal:
                        skipped += 1
                        continue
                    tensor = board_to_tensor_v3(board, flip=flip)
                except Exception:
                    skipped += 1
                    continue
                yield (
                    tensor,
                    _NO_MOVE_HISTORY,
                    np.int64(move_idx),
                    legal,
                    np.float32(1.0),
                    np.float32(values[i]),
                    np.int8(1),
                )
        if skipped:
            print(f'    [dataset worker] skipped {skipped} bad rows',
                  flush=True)


def _make_loader(dataset, batch_size, workers):
    loader_generator = torch.Generator()
    loader_generator.manual_seed(dataset.seed + 1000 * dataset.epoch + 17)
    kwargs = {
        'batch_size': batch_size,
        'num_workers': workers,
        'pin_memory': N.DEVICE.type == 'cuda',
        'collate_fn': make_collate_policy_batch(MOVE_VOCAB),
        'persistent_workers': False,   # each epoch builds a fresh loader
        # Keep worker seeding separate from the resumable model/dropout RNG.
        'generator': loader_generator,
    }
    if workers > 0:
        kwargs['prefetch_factor'] = 6
    return DataLoader(dataset, **kwargs)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data_provenance(data_dir: str,
                          allow_unmanifested: bool = False) -> dict:
    """Validate and fingerprint the corpus contract used by a training run."""
    manifest_path = os.path.join(data_dir, INGEST_MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        if manifest.get('status') != 'complete':
            raise ValueError(
                f'{manifest_path!r} is not a completed ingest manifest'
            )
        manifest_version = int(manifest.get('manifest_version', 1))
        expected = {
            item['name']: {
                'rows': int(item['rows']),
                'sha256': item.get('sha256'),
            }
            for item in manifest.get('outputs', [])
        }
        actual_paths = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
        actual = {
            os.path.basename(path): {
                'rows': pq.ParquetFile(path).metadata.num_rows,
                'sha256': (
                    _sha256_file(path)
                    if manifest_version >= 2 else None
                ),
            }
            for path in actual_paths
        }
        if manifest_version >= 2 and any(
            not entry['sha256'] for entry in expected.values()
        ):
            raise ValueError(
                f'{manifest_path!r} is v2+ but omits an output SHA-256'
            )
        if expected != actual:
            raise ValueError(
                f'data shards do not match the rows/digests recorded in '
                f'{manifest_path!r}; expected {len(expected)} files, found '
                f'{len(actual)}'
            )
        return {
            'kind': 'ingest_manifest',
            'path': os.path.abspath(manifest_path),
            'sha256': _sha256_file(manifest_path),
            'source_revision': manifest.get('source', {}).get('revision'),
        }

    # Legacy stats still provide a stable local corpus fingerprint.
    legacy_path = os.path.join(data_dir, 'ingest_stats.json')
    if os.path.exists(legacy_path):
        with open(legacy_path, encoding='utf-8') as f:
            legacy_stats = json.load(f)
        if 'source_revision' in legacy_stats:
            raise ValueError(
                f'{data_dir!r} contains new-format ingest stats but no completed '
                f'{INGEST_MANIFEST}; refusing a partial ingest'
            )
        print(
            f'WARNING: {data_dir!r} predates {INGEST_MANIFEST}; recording the '
            'legacy ingest_stats.json fingerprint. Re-ingest from a pinned '
            'source revision for fully reproducible training.',
            flush=True,
        )
        return {
            'kind': 'legacy_ingest_stats',
            'path': os.path.abspath(legacy_path),
            'sha256': _sha256_file(legacy_path),
            'source_revision': None,
        }
    if allow_unmanifested:
        print(
            'WARNING: training on explicitly allowed unmanifested data; the '
            'checkpoint cannot identify its source corpus.',
            flush=True,
        )
        return {
            'kind': 'unmanifested_opt_in',
            'path': None,
            'sha256': None,
            'source_revision': None,
        }
    raise FileNotFoundError(
        f'{data_dir!r} has neither {INGEST_MANIFEST} nor ingest_stats.json. '
        'Refusing unproven training data; pass --allow_unmanifested_data only '
        'for an intentional local experiment.'
    )


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            'state': torch.from_numpy(numpy_state[1].copy()),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _runtime_metadata() -> dict:
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_sha': git_sha,
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {
            'torch': str(torch.__version__),
            'numpy': np.__version__,
            'pyarrow': pa.__version__,
            'python_chess': chess.__version__,
        },
        'device': str(N.DEVICE),
    }


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        numpy_state['state'].cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state['position']),
        int(numpy_state['has_gauss']),
        float(numpy_state['cached_gaussian']),
    ))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.cuda.is_available() and state.get('torch_cuda'):
        torch.cuda.set_rng_state_all(
            [rng.cpu() for rng in state['torch_cuda']]
        )


def _save(model, optimizer, scaler, path, meta):
    payload = {
        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_class': type(optimizer).__name__,
        'scaler_state_dict': scaler.state_dict(),
        'rng_state': _capture_rng_state(),
        'board_encoding': BOARD_ENCODING_VERSION_V3,
        'arch_version': 'v4',
        'residual_filters': model.filters,
        'residual_blocks': model.blocks,
        'value_loss_weight': float(os.environ['VALUE_LOSS_WEIGHT']),
        'label_smoothing': float(N.LABEL_SMOOTHING),
        'training_kind': 'lichess_eval_distillation',
        'runtime': _runtime_metadata(),
    }
    payload.update(meta)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _restore_checkpoint(path, model, optimizer, scaler, data_provenance,
                        expected_value_loss_weight: float,
                        expected_label_smoothing: float,
                        expected_training_config: dict | None = None) -> dict:
    """Strictly restore a distillation checkpoint and its training state."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'--resume checkpoint not found: {path}')
    checkpoint = torch.load(
        path, map_location=N.DEVICE, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f'{path!r} is not a checkpoint dictionary')
    required = ('model_state_dict', 'optimizer_state_dict')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'{path!r} is missing resume state: {missing}')
    if checkpoint.get('arch_version', 'v4') != 'v4':
        raise ValueError(f'{path!r} is not a v4 distillation checkpoint')
    if checkpoint.get('board_encoding', BOARD_ENCODING_VERSION_V3) != \
            BOARD_ENCODING_VERSION_V3:
        raise ValueError(f'{path!r} has an incompatible board encoding')
    for key, expected in (
        ('residual_filters', model.filters),
        ('residual_blocks', model.blocks),
    ):
        actual = int(checkpoint.get(key, expected))
        if actual != expected:
            raise ValueError(
                f'{path!r} records {key}={actual}, but this run requested '
                f'{expected}'
            )
    training_kind = checkpoint.get('training_kind')
    if training_kind not in (None, 'lichess_eval_distillation'):
        raise ValueError(
            f'{path!r} has incompatible training_kind={training_kind!r}'
        )
    for key, expected in (
        ('value_loss_weight', expected_value_loss_weight),
        ('label_smoothing', expected_label_smoothing),
    ):
        if key in checkpoint and not np.isclose(
            float(checkpoint[key]), expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f'{path!r} records {key}={checkpoint[key]}, but this run uses '
                f'{expected}; changing the objective is not an exact resume'
            )
    recorded_provenance = checkpoint.get('data_provenance') or {}
    recorded_digest = recorded_provenance.get('sha256')
    current_digest = data_provenance.get('sha256')
    if recorded_digest and recorded_digest != current_digest:
        raise ValueError(
            f'{path!r} was trained on data provenance {recorded_digest}, but '
            f'the current corpus is {current_digest}'
        )
    recorded_config = checkpoint.get('training_config') or {}
    expected_training_config = expected_training_config or {}
    # Reject resume settings that would change scheduling or sample position.
    for key in (
        'lr', 'lr_decay', 'weight_decay', 'batch_size', 'workers', 'seed',
        'channels_last', 'deterministic',
    ):
        if key in recorded_config and key in expected_training_config:
            if recorded_config[key] != expected_training_config[key]:
                raise ValueError(
                    f'{path!r} records {key}={recorded_config[key]!r}, but '
                    f'this run requested {expected_training_config[key]!r}; '
                    'that is not an exact resume'
                )

    if 'rng_state' not in checkpoint:
        # Missing RNG state prevents bit-identical continuation.
        print(
            f'WARNING: {path!r} predates RNG-state capture; the resumed run '
            'is deterministic but not bit-identical to an uninterrupted one.',
            flush=True,
        )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if checkpoint.get('optimizer_class') not in (None, type(optimizer).__name__):
        raise ValueError(
            f'{path!r} used optimizer {checkpoint["optimizer_class"]!r}, not '
            f'{type(optimizer).__name__!r}'
        )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    _restore_rng_state(checkpoint.get('rng_state'))
    return checkpoint


def run_validation(model, val_paths, batch_size, workers, value_criterion,
                   max_batches=None):
    loader = _make_loader(
        DistillShardDataset(val_paths, shuffle=False), batch_size, workers
    )
    return N.evaluate(model, loader, value_criterion, N.DEVICE,
                      max_batches=max_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/distill_chunks_v4')
    parser.add_argument('--output', default='model/grandmaster_resnet_v4_distill.pt')
    parser.add_argument('--filters', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.3,
                        help='LR multiplier applied at each new epoch.')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint_minutes', type=float, default=30,
                        help='Also snapshot the model this often mid-epoch.')
    parser.add_argument('--val_every_minutes', type=float, default=120,
                        help='Run validation (and save the best checkpoint) '
                             'this often mid-epoch. Essential for full-dump '
                             'training where one epoch is ~a day. Mid-epoch '
                             'resume deterministically replays and skips '
                             'already-completed batches.')
    parser.add_argument('--val_batches', type=int, default=120,
                        help='Val batches per evaluation (120 x 1024 ~ 123k '
                             'positions of the held-out shard).')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Debug cap on optimizer steps per epoch.')
    parser.add_argument('--channels_last', action='store_true',
                        help='Use channels_last memory format. Measured 4x '
                             'SLOWER on RTX 3070 for this 8x8 workload, so '
                             'off by default.')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Request deterministic torch algorithms (slower; some CUDA '
             'kernels may be unavailable). Seeds are set regardless.',
    )
    parser.add_argument(
        '--allow_unmanifested_data', action='store_true',
        help='Explicitly allow a corpus with neither ingest_manifest.json nor '
             'legacy ingest_stats.json. Intended only for local experiments.',
    )
    args = parser.parse_args()

    if args.resume and not os.path.exists(args.resume):
        parser.error(f'--resume checkpoint not found: {args.resume}')
    if args.filters < 1 or args.blocks < 1 or args.epochs < 1:
        parser.error('--filters, --blocks, and --epochs must be positive')
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch_size must be positive and --workers non-negative')
    if args.lr <= 0 or not 0 < args.lr_decay <= 1 or args.weight_decay < 0:
        parser.error('require lr > 0, 0 < lr_decay <= 1, and weight_decay >= 0')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max_steps must be positive when provided')
    if args.checkpoint_minutes <= 0 or args.val_every_minutes <= 0:
        parser.error('checkpoint and validation intervals must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    N.assert_training_device()
    N.print_device_info()

    train_paths = glob.glob(os.path.join(args.data_dir, 'train_*.parquet'))
    val_paths = glob.glob(os.path.join(args.data_dir, 'val_*.parquet'))
    if not train_paths or not val_paths:
        raise SystemExit(
            f'{args.data_dir!r} is missing train_/val_ shards. Run '
            'training/ingest_lichess_evals.py first.'
        )
    n_train = sum(pq.ParquetFile(p).metadata.num_rows for p in train_paths)
    print(f'train shards: {len(train_paths)} ({n_train:,} positions) | '
          f'val shards: {len(val_paths)}', flush=True)
    data_provenance = _load_data_provenance(
        args.data_dir, allow_unmanifested=args.allow_unmanifested_data
    )
    print(
        f'data provenance: {data_provenance["kind"]} '
        f'(sha256={data_provenance.get("sha256") or "unavailable"})',
        flush=True,
    )

    model = ChessModelV4(filters=args.filters, blocks=args.blocks).to(N.DEVICE)
    if args.channels_last and N.DEVICE.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=N.DEVICE.type == 'cuda')
    value_criterion = torch.nn.MSELoss(reduction='none')

    start_epoch = 1
    resume_step = 0
    best_val = float('inf')
    if args.resume:
        ckpt = _restore_checkpoint(
            args.resume, model, optimizer, scaler, data_provenance,
            expected_value_loss_weight=float(
                os.environ['VALUE_LOSS_WEIGHT']
            ),
            expected_label_smoothing=float(N.LABEL_SMOOTHING),
            expected_training_config=vars(args),
        )
        checkpoint_epoch = int(ckpt.get('epoch', 0))
        # Legacy checkpoints with a step marker were saved mid-epoch.
        epoch_complete = bool(
            ckpt.get('epoch_complete', 'step' not in ckpt)
        )
        start_epoch = checkpoint_epoch + 1 if epoch_complete else checkpoint_epoch
        resume_step = 0 if epoch_complete else int(ckpt.get('step', 0))
        best_val = ckpt.get('val_loss', float('inf'))
        location = (
            f'after completed epoch {checkpoint_epoch}' if epoch_complete else
            f'at epoch {checkpoint_epoch}, completed step {resume_step:,}'
        )
        print(f'resumed from {args.resume} ({location}; best val '
              f'{best_val:.4f})', flush=True)
        if start_epoch > args.epochs:
            parser.error(
                f'nothing left to train with --epochs {args.epochs}; the '
                f'checkpoint already completed {start_epoch - 1} epoch(s). '
                f'Pass --epochs {start_epoch} to run one more epoch.'
            )
        if args.max_steps is not None and args.max_steps <= resume_step:
            parser.error(
                f'--max_steps {args.max_steps} does not advance beyond the '
                f'resumed step {resume_step}'
            )

    params = sum(p.numel() for p in model.parameters())
    print(f'ChessModelV4 {args.filters}x{args.blocks} | {params:,} params | '
          f'batch {args.batch_size} | VALUE_LOSS_WEIGHT='
          f'{os.environ["VALUE_LOSS_WEIGHT"]}', flush=True)

    train_ds = DistillShardDataset(
        train_paths, shuffle=True, seed=args.seed
    )
    label_smoothing = N.LABEL_SMOOTHING

    training_finished = True
    for epoch in range(start_epoch, args.epochs + 1):
        lr = args.lr * (args.lr_decay ** (epoch - 1))
        for group in optimizer.param_groups:
            group['lr'] = lr
        train_ds.set_epoch(epoch)
        loader = _make_loader(train_ds, args.batch_size, args.workers)
        print(f'\nEpoch {epoch}/{args.epochs}  lr={lr:.2e}', flush=True)

        model.train()
        running = {'loss': 0.0, 'policy': 0.0, 'value': 0.0, 'n': 0,
                   'correct': 0}
        epoch_complete = True
        processed_step = resume_step if epoch == start_epoch else 0
        epoch_start = last_ckpt = last_val = time.monotonic()
        if epoch == start_epoch and resume_step:
            print(
                f'  replaying the deterministic input stream to skip '
                f'{resume_step:,} already-completed batches...',
                flush=True,
            )
        for step, batch in enumerate(loader, start=1):
            if args.max_steps is not None and step > args.max_steps:
                epoch_complete = False
                break
            if epoch == start_epoch and step <= resume_step:
                continue
            boards, moves, move_idx, legal_mask, weights, values, ttypes = batch
            boards = boards.to(N.DEVICE, non_blocking=True)
            if args.channels_last and N.DEVICE.type == 'cuda':
                boards = boards.to(memory_format=torch.channels_last)
            move_idx = move_idx.to(N.DEVICE, non_blocking=True)
            legal_mask = legal_mask.to(N.DEVICE, non_blocking=True)
            values = values.to(N.DEVICE, non_blocking=True)
            ttypes = ttypes.to(N.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(N.DEVICE.type,
                                    enabled=N.DEVICE.type == 'cuda'):
                policy_logits, value_pred = model(boards)
                masked = mask_illegal_logits(policy_logits, legal_mask)
                policy_loss = policy_loss_for_targets(
                    masked, move_idx, ttypes,
                    legal_mask=legal_mask, label_smoothing=label_smoothing,
                ).mean()
                value_loss = value_criterion(
                    value_pred.float(), values.float()
                ).mean()
                loss = policy_loss + float(
                    os.environ['VALUE_LOSS_WEIGHT']) * value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            processed_step = step

            bsz = boards.size(0)
            running['loss'] += float(loss) * bsz
            running['policy'] += float(policy_loss) * bsz
            running['value'] += float(value_loss) * bsz
            running['correct'] += int(
                (masked.argmax(1) == move_idx).sum()
            )
            running['n'] += bsz

            if step % 200 == 0:
                elapsed = time.monotonic() - epoch_start
                pos_s = running['n'] / max(elapsed, 1e-6)
                print(
                    f'  step {step:6,} | loss {running["loss"]/running["n"]:.4f} '
                    f'| policy {running["policy"]/running["n"]:.4f} '
                    f'| value {running["value"]/running["n"]:.4f} '
                    f'| top1 {running["correct"]/running["n"]:.3f} '
                    f'| {pos_s:,.0f} pos/s',
                    flush=True,
                )
            if (time.monotonic() - last_ckpt) >= args.checkpoint_minutes * 60:
                _save(model, optimizer, scaler, args.output + '.midtrain', {
                    'epoch': epoch,
                    'step': step,
                    'epoch_complete': False,
                    'val_loss': best_val,
                    'seed': args.seed,
                    'data_provenance': data_provenance,
                    'training_config': vars(args),
                })
                last_ckpt = time.monotonic()
                print(f'  [midtrain snapshot at step {step:,}]', flush=True)
            if (time.monotonic() - last_val) >= args.val_every_minutes * 60:
                val = run_validation(
                    model, val_paths, args.batch_size,
                    max(args.workers - 3, 1), value_criterion,
                    max_batches=args.val_batches,
                )
                print(
                    f'  [mid-epoch val at step {step:,}] '
                    f'loss {val["loss"]:.4f} | policy {val["policy_loss"]:.4f} '
                    f'| value {val["value_loss"]:.4f} '
                    f'| top1 {val["move_acc"]:.4f}',
                    flush=True,
                )
                if val['loss'] < best_val:
                    best_val = val['loss']
                    _save(model, optimizer, scaler, args.output, {
                        'epoch': epoch,
                        'step': step,
                        'epoch_complete': False,
                        'val_loss': val['loss'],
                        'val_policy_loss': val['policy_loss'],
                        'val_value_loss': val['value_loss'],
                        'val_value_mae': val['value_mae'],
                        'val_move_acc': val['move_acc'],
                        'seed': args.seed,
                        'data_provenance': data_provenance,
                        'training_config': vars(args),
                    })
                    print(f'  saved {args.output} (val {best_val:.4f})',
                          flush=True)
                model.train()   # N.evaluate leaves the model in eval mode
                last_val = time.monotonic()

        print(f'  validating...', flush=True)
        val = run_validation(model, val_paths, args.batch_size,
                             max(args.workers - 3, 1), value_criterion,
                             max_batches=args.val_batches)
        print(
            f'  epoch {epoch}: val loss {val["loss"]:.4f} | '
            f'policy {val["policy_loss"]:.4f} | value {val["value_loss"]:.4f} '
            f'| value_mae {val["value_mae"]:.4f} | top1 {val["move_acc"]:.4f}',
            flush=True,
        )
        if val['loss'] < best_val:
            best_val = val['loss']
            _save(model, optimizer, scaler, args.output, {
                'epoch': epoch,
                'step': processed_step,
                'epoch_complete': epoch_complete,
                'val_loss': val['loss'],
                'val_policy_loss': val['policy_loss'],
                'val_value_loss': val['value_loss'],
                'val_value_mae': val['value_mae'],
                'val_move_acc': val['move_acc'],
                'seed': args.seed,
                'data_provenance': data_provenance,
                'training_config': vars(args),
            })
            print(f'  saved {args.output} (val {best_val:.4f})', flush=True)
        else:
            print(f'  no improvement over {best_val:.4f}; checkpoint kept',
                  flush=True)

        # Keep latest training state separate from the best validation model.
        _save(model, optimizer, scaler, args.output + '.midtrain', {
            'epoch': epoch,
            'step': processed_step,
            'epoch_complete': epoch_complete,
            'val_loss': best_val,
            'seed': args.seed,
            'data_provenance': data_provenance,
            'training_config': vars(args),
        })
        resume_step = 0
        if not epoch_complete:
            training_finished = False
            print(
                f'  stopped at debug cap --max_steps={args.max_steps}; '
                f'{args.output}.midtrain is resumable and the epoch is not '
                'marked complete',
                flush=True,
            )
            break

    print(
        '\ndistillation training complete' if training_finished else
        '\ndistillation training paused at max-step cap',
        flush=True,
    )


if __name__ == '__main__':
    main()
