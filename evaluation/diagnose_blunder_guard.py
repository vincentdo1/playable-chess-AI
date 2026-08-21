"""Compare production and blunder-guarded move quality by game phase.

Positions are rebuilt from FEN and each selected move is scored by Stockfish.
"""

import argparse
import os
import sys
import time

import chess
import chess.engine
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_model import load_trained_model, _get_move_scores  # noqa: E402
from inference.blunder_guard import filter_scored_moves  # noqa: E402
from evaluation.diagnose_value_rerank import (  # noqa: E402
    PHASES, BLUNDER_CP, MATE_SCORE_CP, sample_positions, cp_after_move,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default=os.environ.get('MODEL_PATH',
                                               'model/grandmaster_resnet_v3.pt'))
    parser.add_argument('--chunk_dir', default='data/test_chunks_v3')
    parser.add_argument('--per_phase', type=int, default=150)
    parser.add_argument('--engine_time', type=float, default=0.06)
    parser.add_argument('--stockfish', default=os.environ.get(
        'STOCKFISH_PATH', 'stockfish.exe'))
    parser.add_argument('--value_weight', type=float, default=2.0)
    parser.add_argument('--guard_depth', type=int, default=2)
    parser.add_argument('--guard_margin', type=float, default=150.0)
    parser.add_argument('--guard_candidates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    model = load_trained_model(args.model)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = chess.engine.Limit(time=args.engine_time)

    print("Sampling positions...", flush=True)
    sampled = sample_positions(args.chunk_dir, args.per_phase, args.seed)
    print(f"{len(sampled)} positions sampled\n", flush=True)

    n = len(PHASES)
    stats = {
        name: {'loss': [[] for _ in range(n)],
               'blunders': np.zeros(n, dtype=np.int64)}
        for name in ('production', 'guarded')
    }
    count = np.zeros(n, dtype=np.int64)
    vetoed = np.zeros(n, dtype=np.int64)
    guard_time_total = 0.0
    guard_calls = 0

    try:
        for k, (ph, fen) in enumerate(sampled):
            board = chess.Board(fen)
            if board.is_game_over():
                continue

            scored = _get_move_scores(
                model, board,
                value_weight=args.value_weight, value_candidate_limit=0,
            )
            if not scored:
                continue
            move_prod = scored[0][1]

            t0 = time.perf_counter()
            safe = filter_scored_moves(
                board, scored,
                depth=args.guard_depth,
                margin_cp=args.guard_margin,
                max_candidates=args.guard_candidates,
            )
            guard_time_total += time.perf_counter() - t0
            guard_calls += 1
            move_guard = safe[0][1]

            pov = board.turn
            info = engine.analyse(board, limit)
            best_cp = info['score'].pov(pov).score(mate_score=MATE_SCORE_CP)

            child_cp = {}
            for move in {move_prod, move_guard}:
                child_cp[move] = cp_after_move(engine, board, move, pov, limit)

            count[ph] += 1
            if move_guard != move_prod:
                vetoed[ph] += 1
            for name, move in (('production', move_prod),
                               ('guarded', move_guard)):
                loss = max(0.0, float(best_cp - child_cp[move]))
                loss = min(loss, 2000.0)
                stats[name]['loss'][ph].append(loss)
                if loss >= BLUNDER_CP:
                    stats[name]['blunders'][ph] += 1

            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(sampled)} positions scored", flush=True)
    finally:
        engine.quit()

    print(f"\nGuard: depth={args.guard_depth} margin={args.guard_margin:.0f}cp "
          f"candidates={args.guard_candidates} | avg guard overhead "
          f"{1000 * guard_time_total / max(guard_calls, 1):.0f} ms/move\n")
    header = (
        f"{'phase':>12} | {'n':>5} | {'veto%':>6} | "
        f"{'prod avg':>9} | {'guard avg':>9} | "
        f"{'prod blund%':>11} | {'guard blund%':>12}"
    )
    print(header)
    print('-' * len(header))
    for ph, (name, _, _) in enumerate(PHASES):
        if count[ph] == 0:
            continue
        pr = np.array(stats['production']['loss'][ph])
        gd = np.array(stats['guarded']['loss'][ph])
        print(
            f"{name:>12} | {count[ph]:>5} | "
            f"{vetoed[ph] / count[ph]:>6.1%} | "
            f"{pr.mean():>9.0f} | {gd.mean():>9.0f} | "
            f"{stats['production']['blunders'][ph] / count[ph]:>11.1%} | "
            f"{stats['guarded']['blunders'][ph] / count[ph]:>12.1%}"
        )

    total_pr = np.concatenate([np.array(x) for x in stats['production']['loss']])
    total_gd = np.concatenate([np.array(x) for x in stats['guarded']['loss']])
    print(f"\nOverall avg cp loss: production={total_pr.mean():.0f}  "
          f"guarded={total_gd.mean():.0f}")
    print(f"Overall blunder rate (>={BLUNDER_CP}cp): "
          f"production={(total_pr >= BLUNDER_CP).mean():.1%}  "
          f"guarded={(total_gd >= BLUNDER_CP).mean():.1%}")


if __name__ == '__main__':
    main()
