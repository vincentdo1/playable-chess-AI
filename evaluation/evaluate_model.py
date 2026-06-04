"""Evaluate a trained policy+value model on held-out chunk data."""

import argparse
import glob
import os

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from load_model import _get_move_scores, evaluate_position, load_trained_model
from neural_network import (
    ChunkDataset,
    VALUE_LOSS_WEIGHT,
    collate_policy_batch,
    mask_illegal_logits,
    policy_loss_for_targets,
)


def evaluate_chunks(model, test_dir, batch_size=512, top_k=(1, 3, 5)):
    dataset = ChunkDataset(test_dir, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_policy_batch,
    )
    value_criterion = torch.nn.MSELoss(reduction='none')
    max_k = max(top_k)

    model.eval()
    model_device = next(model.parameters()).device
    total = 0
    total_weight = 0.0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_value_abs_error = 0.0
    positive_weight = 0.0
    negative_total = 0
    correct = {k: 0 for k in top_k}
    weighted_correct = {k: 0.0 for k in top_k}

    with torch.no_grad():
        for boards, moves, targets, legal_mask, sample_weight, value_target, target_type in loader:
            boards = boards.to(model_device)
            moves = moves.to(model_device)
            targets = targets.to(model_device)
            legal_mask = legal_mask.to(model_device)
            sample_weight = sample_weight.to(model_device)
            value_target = value_target.to(model_device)
            target_type = target_type.to(model_device)

            policy_logits, value_pred = model(boards, moves)
            logits = mask_illegal_logits(policy_logits, legal_mask)
            policy_loss_per_sample = policy_loss_for_targets(
                logits, targets, target_type
            )
            value_loss_per_sample = value_criterion(
                value_pred.float(), value_target.float()
            )
            loss_per_sample = (
                policy_loss_per_sample +
                VALUE_LOSS_WEIGHT * value_loss_per_sample
            )
            total_loss += (loss_per_sample * sample_weight).sum().item()
            total_policy_loss += (
                policy_loss_per_sample * sample_weight
            ).sum().item()
            total_value_loss += (
                value_loss_per_sample * sample_weight
            ).sum().item()
            total_value_abs_error += (
                (value_pred.float() - value_target.float()).abs() *
                sample_weight
            ).sum().item()
            total_weight += sample_weight.sum().item()

            predictions = logits.topk(max_k, dim=1).indices
            positive_mask = target_type > 0
            negative_total += (target_type < 0).sum().item()
            positive_weight += (
                sample_weight * positive_mask.float()
            ).sum().item()
            for k in top_k:
                matched = (
                    predictions[:, :k] == targets.unsqueeze(1)
                ).any(dim=1)
                matched = matched & positive_mask
                correct[k] += matched.sum().item()
                weighted_correct[k] += (
                    matched.float() * sample_weight
                ).sum().item()
            total += positive_mask.sum().item()

    metrics = {
        'loss': total_loss / max(total_weight, 1e-6),
        'policy_loss': total_policy_loss / max(total_weight, 1e-6),
        'value_loss': total_value_loss / max(total_weight, 1e-6),
        'value_mae': total_value_abs_error / max(total_weight, 1e-6),
        'positions': total,
        'negative_positions': negative_total,
    }
    metrics.update({f'top_{k}_acc': correct[k] / max(total, 1) for k in top_k})
    metrics.update({
        f'weighted_top_{k}_acc': weighted_correct[k] / max(positive_weight, 1e-6)
        for k in top_k
    })
    return metrics


def print_examples(model, test_dir, max_examples=5, top_k=5):
    if max_examples <= 0:
        return

    shown = 0
    chunk_paths = sorted(glob.glob(os.path.join(test_dir, 'chunk_*.npz')))
    for path in chunk_paths:
        with np.load(path) as data:
            if (
                'fen' not in data or
                'played_uci' not in data or
                'value_target' not in data
            ):
                print("\nRequired example metadata missing; re-run preprocess.py.")
                return

            is_bad_move = (
                data['is_bad_move'] if 'is_bad_move' in data
                else np.zeros(len(data['fen']), dtype=np.int8)
            )
            cp_loss_bucket = (
                data['cp_loss_bucket'] if 'cp_loss_bucket' in data
                else np.full(len(data['fen']), -1, dtype=np.int8)
            )

            for fen, played_uci, cp_loss, value_target, bad_move, bucket in zip(
                data['fen'], data['played_uci'], data['cp_loss'],
                data['value_target'], is_bad_move, cp_loss_bucket
            ):
                board = chess.Board(str(fen))
                scored = _get_move_scores(model, board)[:top_k]
                value_pred = evaluate_position(model, board)
                predicted = [move.uci() for _, move in scored]
                print()
                print(f"FEN       : {fen}")
                print(f"Played    : {played_uci}")
                print(f"CP loss   : {float(cp_loss):.1f}")
                print(f"Bad move  : {bool(bad_move)}  bucket={int(bucket)}")
                print(f"Value     : pred={value_pred:.3f}  target={float(value_target):.3f}")
                print(f"Model top : {', '.join(predicted)}")
                shown += 1
                if shown >= max_examples:
                    return


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained ChessModel on held-out test chunks.'
    )
    parser.add_argument('--model', default=None, help='Path to model checkpoint.')
    parser.add_argument('--test_dir', default=os.environ.get('TEST_DIR', 'data/test_chunks'))
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--examples', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    model = load_trained_model(args.model) if args.model else load_trained_model()
    metrics = evaluate_chunks(model, args.test_dir, batch_size=args.batch_size)

    print("\nHeld-out test metrics")
    print(f"  positions : {metrics['positions']:,} positive")
    if metrics.get('negative_positions'):
        print(f"  negatives : {metrics['negative_positions']:,}")
    print(f"  loss      : {metrics['loss']:.4f}")
    print(f"  policy    : {metrics['policy_loss']:.4f}")
    print(f"  value     : {metrics['value_loss']:.4f}")
    print(f"  value MAE : {metrics['value_mae']:.4f}")
    print(f"  top-1 acc : {metrics['top_1_acc']:.4f}")
    print(f"  top-3 acc : {metrics['top_3_acc']:.4f}")
    print(f"  top-5 acc : {metrics['top_5_acc']:.4f}")
    print(f"  weighted top-1 acc : {metrics['weighted_top_1_acc']:.4f}")
    print(f"  weighted top-3 acc : {metrics['weighted_top_3_acc']:.4f}")
    print(f"  weighted top-5 acc : {metrics['weighted_top_5_acc']:.4f}")

    print_examples(model, args.test_dir, args.examples, args.top_k)


if __name__ == '__main__':
    main()
