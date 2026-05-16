"""Evaluate a trained policy model on held-out chunk data."""

import argparse
import glob
import os

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader

from load_model import _get_move_scores, load_trained_model
from neural_network import (
    ChunkDataset,
    collate_policy_batch,
    mask_illegal_logits,
)


def evaluate_chunks(model, test_dir, batch_size=512, top_k=(1, 3, 5)):
    dataset = ChunkDataset(test_dir, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_policy_batch,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    max_k = max(top_k)

    model.eval()
    model_device = next(model.parameters()).device
    total = 0
    total_weight = 0.0
    total_loss = 0.0
    correct = {k: 0 for k in top_k}
    weighted_correct = {k: 0.0 for k in top_k}

    with torch.no_grad():
        for boards, moves, targets, legal_mask, sample_weight in loader:
            boards = boards.to(model_device)
            moves = moves.to(model_device)
            targets = targets.to(model_device)
            legal_mask = legal_mask.to(model_device)
            sample_weight = sample_weight.to(model_device)

            logits = mask_illegal_logits(model(boards, moves), legal_mask)
            loss_per_sample = criterion(logits, targets)
            total_loss += (loss_per_sample * sample_weight).sum().item()
            total_weight += sample_weight.sum().item()

            predictions = logits.topk(max_k, dim=1).indices
            for k in top_k:
                matched = (
                    predictions[:, :k] == targets.unsqueeze(1)
                ).any(dim=1)
                correct[k] += matched.sum().item()
                weighted_correct[k] += (
                    matched.float() * sample_weight
                ).sum().item()
            total += targets.numel()

    metrics = {
        'loss': total_loss / max(total_weight, 1e-6),
        'positions': total,
    }
    metrics.update({f'top_{k}_acc': correct[k] / total for k in top_k})
    metrics.update({
        f'weighted_top_{k}_acc': weighted_correct[k] / max(total_weight, 1e-6)
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
            if 'fen' not in data or 'played_uci' not in data:
                print("\nNo FEN metadata found in chunks; re-run preprocess.py.")
                return

            for fen, played_uci, cp_loss in zip(
                data['fen'], data['played_uci'], data['cp_loss']
            ):
                board = chess.Board(str(fen))
                scored = _get_move_scores(model, board)[:top_k]
                predicted = [move.uci() for _, move in scored]
                print()
                print(f"FEN       : {fen}")
                print(f"Played    : {played_uci}")
                print(f"CP loss   : {float(cp_loss):.1f}")
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
    print(f"  positions : {metrics['positions']:,}")
    print(f"  loss      : {metrics['loss']:.4f}")
    print(f"  top-1 acc : {metrics['top_1_acc']:.4f}")
    print(f"  top-3 acc : {metrics['top_3_acc']:.4f}")
    print(f"  top-5 acc : {metrics['top_5_acc']:.4f}")
    print(f"  weighted top-1 acc : {metrics['weighted_top_1_acc']:.4f}")
    print(f"  weighted top-3 acc : {metrics['weighted_top_3_acc']:.4f}")
    print(f"  weighted top-5 acc : {metrics['weighted_top_5_acc']:.4f}")

    print_examples(model, args.test_dir, args.examples, args.top_k)


if __name__ == '__main__':
    main()
