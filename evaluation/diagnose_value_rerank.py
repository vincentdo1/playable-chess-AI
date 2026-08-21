"""Compare policy-only and value-reranked move quality by game phase.

Positions are rebuilt from FEN and each selected move is scored by Stockfish.
"""

import argparse
import glob
import os
import sys

import chess
import chess.engine
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_model import load_trained_model, _get_move_scores  # noqa: E402

PHASES = [
    ('moves 1-10', 1, 10),
    ('moves 11-20', 11, 20),
    ('moves 21-30', 21, 30),
    ('moves 31-40', 31, 40),
    ('moves 41+', 41, 10_000),
]
MATE_SCORE_CP = 100_000
BLUNDER_CP = 300


def phase_index(fullmove: int) -> int:
    for i, (_, lo, hi) in enumerate(PHASES):
        if lo <= fullmove <= hi:
            return i
    return len(PHASES) - 1


def sample_positions(chunk_dir, per_phase, seed):
    """Stratified sample of (fen, played_uci) positives per phase bucket."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in PHASES]
    paths = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))
    if not paths:
        raise SystemExit(f"No chunks in {chunk_dir!r}")
    for path in paths:
        with np.load(path) as data:
            fens = data['fen']
            target_type = data['policy_target_type']
        for i in np.where(target_type > 0)[0]:
            fen = str(fens[i])
            buckets[phase_index(int(fen.rsplit(' ', 1)[1]))].append(fen)
    sampled = []
    for ph, bucket in enumerate(buckets):
        take = min(per_phase, len(bucket))
        for j in rng.choice(len(bucket), size=take, replace=False):
            sampled.append((ph, bucket[j]))
    rng.shuffle(sampled)
    return sampled


def cp_after_move(engine, board, move, pov_color, limit):
    board.push(move)
    try:
        if board.is_checkmate():
            return MATE_SCORE_CP
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        info = engine.analyse(board, limit)
        return info['score'].pov(pov_color).score(mate_score=MATE_SCORE_CP)
    finally:
        board.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default=os.environ.get('MODEL_PATH',
                                               'model/grandmaster_resnet_v3.pt'))
    parser.add_argument('--chunk_dir', default='data/test_chunks_v3')
    parser.add_argument('--per_phase', type=int, default=200)
    parser.add_argument('--engine_time', type=float, default=0.06)
    parser.add_argument('--stockfish', default=os.environ.get(
        'STOCKFISH_PATH', 'stockfish.exe'))
    parser.add_argument('--value_weight', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    model = load_trained_model(args.model)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = chess.engine.Limit(time=args.engine_time)

    print("Sampling positions...", flush=True)
    sampled = sample_positions(args.chunk_dir, args.per_phase, args.seed)
    print(f"{len(sampled)} positions sampled "
          f"({args.per_phase} per phase target)\n", flush=True)

    n = len(PHASES)
    stats = {
        'policy': {'loss': [[] for _ in range(n)],
                   'blunders': np.zeros(n, dtype=np.int64)},
        'production': {'loss': [[] for _ in range(n)],
                       'blunders': np.zeros(n, dtype=np.int64)},
    }
    count = np.zeros(n, dtype=np.int64)
    differed = np.zeros(n, dtype=np.int64)

    try:
        for k, (ph, fen) in enumerate(sampled):
            board = chess.Board(fen)  # empty move stack — same as /api/move
            if board.is_game_over():
                continue

            scored_policy = _get_move_scores(model, board, value_weight=0.0)
            scored_prod = _get_move_scores(
                model, board,
                value_weight=args.value_weight, value_candidate_limit=0,
            )
            if not scored_policy or not scored_prod:
                continue
            move_policy = scored_policy[0][1]
            move_prod = scored_prod[0][1]

            pov = board.turn
            info = engine.analyse(board, limit)
            best_cp = info['score'].pov(pov).score(mate_score=MATE_SCORE_CP)

            child_cp = {}
            for move in {move_policy, move_prod}:
                child_cp[move] = cp_after_move(engine, board, move, pov, limit)

            count[ph] += 1
            if move_policy != move_prod:
                differed[ph] += 1
            for name, move in (('policy', move_policy),
                               ('production', move_prod)):
                loss = max(0.0, float(best_cp - child_cp[move]))
                loss = min(loss, 2000.0)  # cap mate-vs-eval artifacts
                stats[name]['loss'][ph].append(loss)
                if loss >= BLUNDER_CP:
                    stats[name]['blunders'][ph] += 1

            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(sampled)} positions scored", flush=True)
    finally:
        engine.quit()

    print(f"\nStockfish cp-loss of the chosen move ({args.engine_time:.2f}s "
          f"per eval), production = value_weight={args.value_weight}, "
          "value_candidates=0 (all):\n")
    header = (
        f"{'phase':>12} | {'n':>5} | {'diff%':>6} | "
        f"{'policy avg':>10} | {'prod avg':>9} | "
        f"{'policy med':>10} | {'prod med':>8} | "
        f"{'policy blund%':>13} | {'prod blund%':>11}"
    )
    print(header)
    print('-' * len(header))
    for ph, (name, _, _) in enumerate(PHASES):
        if count[ph] == 0:
            continue
        pl = np.array(stats['policy']['loss'][ph])
        pr = np.array(stats['production']['loss'][ph])
        print(
            f"{name:>12} | {count[ph]:>5} | "
            f"{differed[ph] / count[ph]:>6.1%} | "
            f"{pl.mean():>10.0f} | {pr.mean():>9.0f} | "
            f"{np.median(pl):>10.0f} | {np.median(pr):>8.0f} | "
            f"{stats['policy']['blunders'][ph] / count[ph]:>13.1%} | "
            f"{stats['production']['blunders'][ph] / count[ph]:>11.1%}"
        )

    total_pl = np.concatenate([np.array(x) for x in stats['policy']['loss']])
    total_pr = np.concatenate(
        [np.array(x) for x in stats['production']['loss']])
    print(f"\nOverall avg cp loss: policy-only={total_pl.mean():.0f}  "
          f"production={total_pr.mean():.0f}")
    print(f"Overall blunder rate (>={BLUNDER_CP}cp): "
          f"policy-only={(total_pl >= BLUNDER_CP).mean():.1%}  "
          f"production={(total_pr >= BLUNDER_CP).mean():.1%}")


if __name__ == '__main__':
    main()
