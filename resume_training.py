"""Resume/extend supervised training from an existing checkpoint, safely.

Why this exists
---------------
The base model `grandmaster_model_perspective_resnet_negatives_v2.pt` stopped
improving at epoch 6 (val_loss=1.832) and early-stopping killed the run at
epoch ~11. Inspection of neural_network.py shows the likely cause is a schedule
conflict, not true convergence:

  * ReduceLROnPlateau(factor=0.5, patience=3) only halved the LR once before
  * EARLY_STOP_PATIENCE=5 terminated the run two epochs later.

So the reduced learning rate never had time to take effect. This script resumes
from the saved checkpoint with a gentler, non-self-sabotaging schedule and more
patience, to squeeze out the improvement that was left on the table.

Safety guarantees (so the result is never worse than what you have now)
----------------------------------------------------------------------
1. Reads the loaded checkpoint's recorded val_loss and seeds best_val_loss with
   it, so a new checkpoint is ONLY written when it genuinely beats the current
   model on validation. The current model can never be regressed by this run.
2. Writes to a SEPARATE --output path. The input checkpoint file is never
   touched, so your deployed model is safe no matter what happens.
3. Re-validates the loaded weights up front and prints that baseline val_loss,
   so you can confirm the resume actually started from your model (sanity check
   against silently training from random init).

It reuses neural_network.py's dataset, collate, model, loss and per-epoch
train/eval functions verbatim, so the math matches the original training
exactly. Only the optimizer/scheduler/early-stop policy and the checkpoint
gating differ.

Usage (PowerShell, venv active, run from repo root):
  $env:TRAIN_DIR = "data/train_chunks"
  $env:VAL_DIR   = "data/val_chunks"
  python resume_training.py `
    --init model/grandmaster_model_perspective_resnet_negatives_v2.pt `
    --output model/grandmaster_resnet_v2_resumed.pt `
    --epochs 40 --lr 3e-4 --early_stop_patience 10 --lr_patience 4

Then compare the two with eval_arena.py before deploying the new one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Force line-buffered stdout/stderr so `Tee-Object` / `tail -f` show progress in
# real time even when the user forgets PYTHONUNBUFFERED=1. Belt-and-suspenders
# with the flush=True we use on key prints below.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neural_network as N


def _build_loaders():
    train_ds = N.ChunkDataset(N.TRAIN_DIR, shuffle=True)
    val_ds = N.ChunkDataset(N.VAL_DIR, shuffle=False)
    train_loader = DataLoader(
        train_ds, batch_size=N.BATCH_SIZE, num_workers=N.TRAIN_NUM_WORKERS,
        pin_memory=N.DEVICE.type == 'cuda',
        persistent_workers=N.TRAIN_NUM_WORKERS > 0,
        collate_fn=N.collate_policy_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=N.BATCH_SIZE, num_workers=N.VAL_NUM_WORKERS,
        pin_memory=N.DEVICE.type == 'cuda',
        persistent_workers=N.VAL_NUM_WORKERS > 0,
        collate_fn=N.collate_policy_batch,
    )
    return train_loader, val_loader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--init', required=True,
                        help='Checkpoint to resume FROM (never modified).')
    parser.add_argument('--output', required=True,
                        help='Where to write the improved checkpoint (must differ '
                             'from --init).')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Max additional epochs.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Resume LR. Lower than the original 1e-3 because we '
                             'are fine-tuning a partly-trained model, not starting '
                             'cold.')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Epochs without val improvement before stopping. '
                             'Higher than the original 5 so the reduced LR can '
                             'actually take effect.')
    parser.add_argument('--lr_patience', type=int, default=4,
                        help='Plateau patience before halving LR. Kept strictly '
                             'below early_stop_patience so LR drops BEFORE the run '
                             'is allowed to terminate.')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()

    if os.path.abspath(args.init) == os.path.abspath(args.output):
        raise SystemExit("--output must differ from --init so the source model "
                         "is never overwritten.")
    if args.lr_patience >= args.early_stop_patience:
        raise SystemExit("--lr_patience must be < --early_stop_patience, otherwise "
                         "early stopping fires before the LR is ever reduced "
                         "(the exact bug that stalled the original run).")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = N.DEVICE
    # Use the patched neural_network preflight if present (REQUIRE_CUDA guard +
    # device banner). Guarded so this still runs against an unpatched copy.
    if hasattr(N, 'assert_training_device'):
        N.assert_training_device()
    if hasattr(N, 'print_device_info'):
        N.print_device_info()
    else:
        print(f"Device: {device}", flush=True)
    if device.type != 'cuda':
        print("  WARNING: not on CUDA; full-dataset training will be very slow.",
              flush=True)

    # --- Load checkpoint (weights + recorded val_loss) ---
    if not os.path.exists(args.init):
        raise SystemExit(f"--init checkpoint not found: {args.init}")
    checkpoint = torch.load(args.init, map_location=device, weights_only=False)

    model = N.ChessModel().to(device)
    load_result, skipped = N.load_compatible_state_dict(model, checkpoint)
    if load_result.missing_keys:
        print(f"  Newly initialized (were missing): {load_result.missing_keys}")
    if skipped:
        print(f"  Skipped incompatible tensors: {skipped}")
    if load_result.missing_keys or skipped:
        # If big chunks of the net are randomly initialized, resuming is not
        # actually resuming. Refuse rather than silently train from rubble.
        raise SystemExit(
            "Checkpoint does not fully match the current architecture; aborting "
            "so we do not accidentally train from partial/random weights."
        )

    recorded_val = checkpoint.get('val_loss')
    print(f"Loaded {args.init}")
    print(f"  Recorded val_loss in checkpoint: {recorded_val}")

    train_loader, val_loader = _build_loaders()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    value_criterion = nn.MSELoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.lr_patience, min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    # --- Baseline: re-validate the loaded weights so we KNOW our starting point ---
    print("\nRe-validating loaded weights to establish the baseline...")
    baseline = N.evaluate(model, val_loader, value_criterion, device,
                          max_batches=N.MAX_VAL_BATCHES)
    baseline_val = baseline['loss']
    print(f"  Baseline val_loss (recomputed): {baseline_val:.4f}  "
          f"move_acc={baseline['move_acc']:.4f}")

    # Gate against the BETTER of recorded and recomputed, so we never regress.
    best_val_loss = baseline_val
    if recorded_val is not None:
        best_val_loss = min(best_val_loss, float(recorded_val))
    print(f"  New checkpoints will only be saved if val_loss < {best_val_loss:.4f}")

    print(f"\nResume config: lr={args.lr}  max_epochs={args.epochs}  "
          f"lr_patience={args.lr_patience}  early_stop={args.early_stop_patience}")
    print(f"Output (separate file): {args.output}\n")

    patience_count = 0
    improved_ever = False
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        train_metrics = N.train_one_epoch(
            model, train_loader, optimizer, scaler, value_criterion, device,
            max_batches=N.MAX_TRAIN_BATCHES,
        )
        val_metrics = N.evaluate(
            model, val_loader, value_criterion, device,
            max_batches=N.MAX_VAL_BATCHES,
        )
        print(
            f"  train loss={train_metrics['loss']:.4f} "
            f"policy={train_metrics['policy_loss']:.4f} "
            f"value={train_metrics['value_loss']:.4f} "
            f"move_acc={train_metrics['move_acc']:.4f}"
        )
        print(
            f"  val   loss={val_metrics['loss']:.4f} "
            f"policy={val_metrics['policy_loss']:.4f} "
            f"value={val_metrics['value_loss']:.4f} "
            f"move_acc={val_metrics['move_acc']:.4f}  ({time.time()-t0:.0f}s)"
        )
        scheduler.step(val_metrics['loss'])

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_count = 0
            improved_ever = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_policy_loss': val_metrics['policy_loss'],
                'val_value_loss': val_metrics['value_loss'],
                'val_value_mae': val_metrics['value_mae'],
                'board_encoding': N.BOARD_ENCODING_VERSION,
                'residual_filters': N.RESIDUAL_FILTERS,
                'residual_blocks': N.RESIDUAL_BLOCKS,
                'value_loss_weight': N.VALUE_LOSS_WEIGHT,
                'negative_policy_loss_weight': N.NEGATIVE_POLICY_LOSS_WEIGHT,
                'resumed_from': args.init,
            }, args.output)
            print(f"  -> improved; saved {args.output} (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  no improvement ({patience_count}/{args.early_stop_patience})")
            if patience_count >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print()
    if improved_ever:
        print(f"Done. Improved model saved to {args.output} "
              f"(val_loss {best_val_loss:.4f} < baseline {baseline_val:.4f}).")
        print("Next: compare it against the current model before deploying, e.g.")
        print(f"  python eval_arena.py --model_a {args.output} "
              f"--model_b {args.init} --method_a mcts --method_b mcts --games 20")
    else:
        print("Done. No epoch beat the baseline; --output was NOT written.")
        print("Your current model is already at/near this schedule's floor. "
              "Consider a different lr, more data, or accept the current model.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
