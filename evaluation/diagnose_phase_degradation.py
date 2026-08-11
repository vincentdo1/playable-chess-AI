"""Diagnose why the model plays worse as the game goes on.

Buckets held-out test positions by fullmove number and reports, per bucket:

  - top-1 policy accuracy and policy NLL with the TRUE move history
    (what eval_arena / play_match see), versus with an ALL-ZERO move
    history (what the web backend actually feeds the LSTM, because
    /api/move rebuilds the board from FEN and the move stack is empty)
  - how often the argmax move changes between the two history inputs
  - value-head MAE and correlation against the Stockfish-derived target
  - mean policy entropy over legal moves (a flat policy means the
    value-head reranking in load_model.predict_next_move dominates)

Usage:
  python -m evaluation.diagnose_phase_degradation \
      --model model/grandmaster_resnet_v2_resumed.pt \
      --chunk_dir data/test_chunks --max_positions 150000
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import MOVE_VOCAB_SIZE, mask_illegal_logits  # noqa: E402
from load_model import load_trained_model  # noqa: E402

PHASES = [
    ('moves 1-10', 1, 10),
    ('moves 11-20', 11, 20),
    ('moves 21-30', 21, 30),
    ('moves 31-40', 31, 40),
    ('moves 41+', 41, 10_000),
]


def fullmove_from_fen(fen: str) -> int:
    return int(fen.rsplit(' ', 1)[1])


def phase_index(fullmove: int) -> int:
    for i, (_, lo, hi) in enumerate(PHASES):
        if lo <= fullmove <= hi:
            return i
    return len(PHASES) - 1


class Accumulator:
    def __init__(self):
        n = len(PHASES)
        self.count = np.zeros(n, dtype=np.int64)
        self.top1_true = np.zeros(n, dtype=np.int64)
        self.top1_zero = np.zeros(n, dtype=np.int64)
        self.nll_true = np.zeros(n, dtype=np.float64)
        self.nll_zero = np.zeros(n, dtype=np.float64)
        self.argmax_changed = np.zeros(n, dtype=np.int64)
        self.value_abs_err = np.zeros(n, dtype=np.float64)
        self.value_pred_sum = np.zeros(n, dtype=np.float64)
        self.entropy = np.zeros(n, dtype=np.float64)
        # for correlation
        self.vp = [[] for _ in range(n)]
        self.vt = [[] for _ in range(n)]


@torch.no_grad()
def run(model, device, chunk_paths, max_positions, batch_size):
    acc = Accumulator()
    seen = 0
    for path in chunk_paths:
        if seen >= max_positions:
            break
        with np.load(path) as data:
            boards = data['boards']
            moves = data['moves']
            move_idx = data['move_idx']
            fens = data['fen']
            target_type = data['policy_target_type']
            value_target = data['value_target']
            legal_indices = data['legal_move_indices']
            legal_offsets = data['legal_move_offsets']

        keep = np.where(target_type > 0)[0]
        for start in range(0, len(keep), batch_size):
            if seen >= max_positions:
                break
            idx = keep[start:start + batch_size]
            b = boards[idx]
            m = moves[idx]
            mi = move_idx[idx].astype(np.int64)
            vt = value_target[idx]
            phases = np.array(
                [phase_index(fullmove_from_fen(str(f))) for f in fens[idx]]
            )

            legal_mask = torch.zeros(
                (len(idx), MOVE_VOCAB_SIZE), dtype=torch.bool
            )
            for row, i in enumerate(idx):
                s, e = legal_offsets[i], legal_offsets[i + 1]
                legal_mask[row, torch.as_tensor(
                    legal_indices[s:e].astype(np.int64))] = True

            board_t = (
                torch.from_numpy(b).float().permute(0, 3, 1, 2)
                .contiguous().to(device)
            )
            move_t = torch.from_numpy(m).float().to(device)
            zero_t = torch.zeros_like(move_t)
            legal_mask = legal_mask.to(device)
            mi_t = torch.from_numpy(mi).to(device)

            logits_true, value_true = model(board_t, move_t)
            logits_zero, _ = model(board_t, zero_t)
            lt = mask_illegal_logits(logits_true.float(), legal_mask)
            lz = mask_illegal_logits(logits_zero.float(), legal_mask)
            logp_t = torch.log_softmax(lt, dim=1)
            logp_z = torch.log_softmax(lz, dim=1)

            nll_t = -logp_t.gather(1, mi_t.unsqueeze(1)).squeeze(1)
            nll_z = -logp_z.gather(1, mi_t.unsqueeze(1)).squeeze(1)
            am_t = lt.argmax(1)
            am_z = lz.argmax(1)
            p_t = logp_t.exp()
            ent = -(p_t * logp_t.clamp_min(-30)).sum(1)

            top1_t = (am_t == mi_t).cpu().numpy()
            top1_z = (am_z == mi_t).cpu().numpy()
            changed = (am_t != am_z).cpu().numpy()
            nll_t = nll_t.cpu().numpy()
            nll_z = nll_z.cpu().numpy()
            ent = ent.cpu().numpy()
            vp = value_true.float().cpu().numpy()

            for ph in range(len(PHASES)):
                sel = phases == ph
                if not sel.any():
                    continue
                acc.count[ph] += sel.sum()
                acc.top1_true[ph] += top1_t[sel].sum()
                acc.top1_zero[ph] += top1_z[sel].sum()
                acc.nll_true[ph] += nll_t[sel].sum()
                acc.nll_zero[ph] += nll_z[sel].sum()
                acc.argmax_changed[ph] += changed[sel].sum()
                acc.value_abs_err[ph] += np.abs(vp[sel] - vt[sel]).sum()
                acc.value_pred_sum[ph] += vp[sel].sum()
                acc.entropy[ph] += ent[sel].sum()
                acc.vp[ph].extend(vp[sel].tolist())
                acc.vt[ph].extend(vt[sel].tolist())

            seen += len(idx)
        print(f"  processed {seen:,} positions ({os.path.basename(path)})",
              flush=True)
    return acc, seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default=os.environ.get('MODEL_PATH',
                                               'model/grandmaster_resnet_v2_resumed.pt'))
    parser.add_argument('--chunk_dir', default='data/test_chunks')
    parser.add_argument('--max_positions', type=int, default=150_000)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    chunk_paths = sorted(glob.glob(os.path.join(args.chunk_dir, 'chunk_*.npz')))
    if not chunk_paths:
        raise SystemExit(f"No chunks in {args.chunk_dir!r}")

    model = load_trained_model(args.model)
    device = next(model.parameters()).device
    print(f"Diagnosing {args.model} on {args.chunk_dir} "
          f"(up to {args.max_positions:,} positions)\n", flush=True)

    acc, seen = run(model, device, chunk_paths, args.max_positions,
                    args.batch_size)

    print(f"\n{seen:,} positive test positions analyzed.")
    print("\nPer-phase metrics "
          "(true history = eval-arena conditions, zero history = live web "
          "backend conditions):\n")
    header = (
        f"{'phase':>12} | {'n':>7} | {'top1 true':>9} | {'top1 zero':>9} | "
        f"{'nll true':>8} | {'nll zero':>8} | {'am-diff':>8} | "
        f"{'val MAE':>7} | {'val corr':>8} | {'entropy':>7}"
    )
    print(header)
    print('-' * len(header))
    for ph, (name, _, _) in enumerate(PHASES):
        n = acc.count[ph]
        if n == 0:
            continue
        vp = np.array(acc.vp[ph])
        vt = np.array(acc.vt[ph])
        corr = (np.corrcoef(vp, vt)[0, 1]
                if len(vp) > 2 and vp.std() > 1e-9 and vt.std() > 1e-9
                else float('nan'))
        print(
            f"{name:>12} | {n:>7,} | "
            f"{acc.top1_true[ph] / n:>9.3f} | {acc.top1_zero[ph] / n:>9.3f} | "
            f"{acc.nll_true[ph] / n:>8.3f} | {acc.nll_zero[ph] / n:>8.3f} | "
            f"{acc.argmax_changed[ph] / n:>8.3f} | "
            f"{acc.value_abs_err[ph] / n:>7.3f} | {corr:>8.3f} | "
            f"{acc.entropy[ph] / n:>7.3f}"
        )

    print(
        "\nReading the table:\n"
        "  - top1 true falling with phase = supervised generalization "
        "limit (overfitting / data coverage).\n"
        "  - top1 zero well below top1 true = the live backend is "
        "crippled by the empty move stack (FEN-only /api/move).\n"
        "  - am-diff = fraction of positions where the served move "
        "differs from the evaluated move purely due to missing history.\n"
        "  - val MAE rising + val corr falling with phase = the "
        "value_weight=2.0 reranker amplifies noise late in the game.\n"
        "  - entropy rising with phase = policy gets flat, so value "
        "reranking increasingly decides the move."
    )


if __name__ == '__main__':
    main()
