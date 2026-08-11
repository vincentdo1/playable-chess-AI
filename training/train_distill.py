"""Phase-1 distillation trainer: v4 SE-ResNet on Lichess-eval shards.

Trains ChessModelV4 on the compact (fen, move, value_target, depth) shards
built by training/ingest_lichess_evals.py. Board tensors and legal masks are
computed on the fly in DataLoader workers with the audited v3 codecs, so the
training inputs are byte-identical to what serving builds from a FEN
(docs/ROADMAP_2500.md WS1/WS2).

Checkpoints store arch_version='v4' (+ filters/blocks), which
load_model.load_trained_model dispatches on; the board encoding remains
'perspective_v3'.

Usage (defaults = Phase-1 recipe):
  python -m training.train_distill                # 3 epochs over data/distill_chunks_v4
  python -m training.train_distill --resume model/grandmaster_resnet_v4_distill.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import time

# pyarrow MUST be imported before torch: with torch 2.6 + pyarrow 24 on
# Windows the reverse order segfaults on first parquet access (DLL clash).
# DataLoader spawn workers re-import this module, so keeping the order here
# protects them too.
import pyarrow.parquet as pq

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# Distilled value labels are dense engine evals, so the value head deserves
# more weight than the 0.25 used against noisy game outcomes. Must be set
# before neural_network reads its env config.
os.environ.setdefault('VALUE_LOSS_WEIGHT', '1.0')

import neural_network as N
from neural_network import (
    BOARD_ENCODING_VERSION_V3, ChessModelV4, _NO_MOVE_HISTORY,
    board_to_tensor_v3, legal_policy_indices_v3, make_collate_policy_batch,
    mask_illegal_logits, move_to_policy_index_v3, policy_loss_for_targets,
)

MOVE_VOCAB = N.MOVE_VOCAB_SIZE_V3


class DistillShardDataset(IterableDataset):
    """Streams (fen, move, value) shards; encodes tensors in the workers.

    Rows whose stored move is not legal in the stored FEN are skipped and
    counted instead of raising — a handful of bad rows must not kill an
    overnight run. (Ingestion-time sampling puts the bad-row rate at ~0.)
    """

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
                    # parse_uci (not Move.from_uci): Lichess PVs write castling
                    # as king-takes-rook (e1h1); parse_uci normalizes it to
                    # standard UCI and validates legality.
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
    kwargs = {
        'batch_size': batch_size,
        'num_workers': workers,
        'pin_memory': N.DEVICE.type == 'cuda',
        'collate_fn': make_collate_policy_batch(MOVE_VOCAB),
        'persistent_workers': False,   # each epoch builds a fresh loader
    }
    if workers > 0:
        kwargs['prefetch_factor'] = 6
    return DataLoader(dataset, **kwargs)


def _save(model, optimizer, path, meta):
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'board_encoding': BOARD_ENCODING_VERSION_V3,
        'arch_version': 'v4',
        'residual_filters': model.filters,
        'residual_blocks': model.blocks,
        'value_loss_weight': float(os.environ['VALUE_LOSS_WEIGHT']),
        'training_kind': 'lichess_eval_distillation',
    }
    payload.update(meta)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
                             'training where one epoch is ~a day. Note: '
                             '--resume continues with the NEXT epoch — a '
                             'partially trained epoch counts as complete '
                             '(the stream cannot seek back to where it died).')
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
    args = parser.parse_args()

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

    model = ChessModelV4(filters=args.filters, blocks=args.blocks).to(N.DEVICE)
    if args.channels_last and N.DEVICE.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=N.DEVICE.type == 'cuda')
    value_criterion = torch.nn.MSELoss(reduction='none')

    start_epoch = 1
    best_val = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=N.DEVICE,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('val_loss', float('inf'))
        print(f'resumed from {args.resume} (epoch {start_epoch - 1} counted '
              f'as complete, best val {best_val:.4f})', flush=True)
        if start_epoch > args.epochs:
            print(f'NOTE: nothing left to train with --epochs {args.epochs} '
                  f'(checkpoint already counts {start_epoch - 1} epoch(s)). '
                  f'Pass --epochs {start_epoch} to run one more epoch.',
                  flush=True)

    params = sum(p.numel() for p in model.parameters())
    print(f'ChessModelV4 {args.filters}x{args.blocks} | {params:,} params | '
          f'batch {args.batch_size} | VALUE_LOSS_WEIGHT='
          f'{os.environ["VALUE_LOSS_WEIGHT"]}', flush=True)

    train_ds = DistillShardDataset(train_paths, shuffle=True)
    label_smoothing = N.LABEL_SMOOTHING

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
        epoch_start = last_ckpt = last_val = time.monotonic()
        for step, batch in enumerate(loader, start=1):
            if args.max_steps is not None and step > args.max_steps:
                break
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
                _save(model, optimizer, args.output + '.midtrain',
                      {'epoch': epoch, 'step': step, 'val_loss': best_val})
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
                    _save(model, optimizer, args.output, {
                        'epoch': epoch,
                        'step': step,
                        'val_loss': val['loss'],
                        'val_policy_loss': val['policy_loss'],
                        'val_value_loss': val['value_loss'],
                        'val_value_mae': val['value_mae'],
                        'val_move_acc': val['move_acc'],
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
            _save(model, optimizer, args.output, {
                'epoch': epoch,
                'val_loss': val['loss'],
                'val_policy_loss': val['policy_loss'],
                'val_value_loss': val['value_loss'],
                'val_value_mae': val['value_mae'],
                'val_move_acc': val['move_acc'],
            })
            print(f'  saved {args.output} (val {best_val:.4f})', flush=True)
        else:
            print(f'  no improvement over {best_val:.4f}; checkpoint kept',
                  flush=True)

    print('\ndistillation training complete', flush=True)


if __name__ == '__main__':
    main()
