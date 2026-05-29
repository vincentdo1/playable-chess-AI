"""Head-to-head evaluation between two checkpoints, with an Elo estimate.

The self-play loop only tells you the training loss went down; it does NOT tell
you the model got *stronger*. The only honest measure of strength is games. This
harness loads two checkpoints (e.g. a self-play iteration vs the supervised base),
plays an even-colored match, and reports the score plus an Elo difference with a
95% confidence interval.

Each side may move via:
  - mcts   : MCTS/PUCT (the strong setting; matches how you'd deploy)
  - policy : single forward pass, temperature sampling (raw policy)
  - search : alpha-beta with the network as evaluator

Colors are alternated every game so neither model gets a first-move advantage,
and a fixed seed makes a run reproducible. Use a non-zero temperature (policy) or
the natural MCTS move sampling to avoid every game being identical; for the
sharpest strength signal, play deterministically and rely on color alternation
plus Dirichlet-free search for variety, or bump --games.
"""

from __future__ import annotations

import argparse
import math
import random

import chess
import numpy as np

from load_model import load_trained_model, predict_next_move
from search_player import search_best_move
from mcts_player import mcts_search_best_move


def pick_move(method, model, board, args):
    """Return a chess.Move for the given model using the chosen method."""
    if method == 'mcts':
        return mcts_search_best_move(
            model, board,
            num_simulations=args.sims,
            c_puct=args.c_puct,
            batch_size=args.mcts_batch_size,
            policy_temperature=args.policy_temperature,
            move_temperature=args.move_temperature,
        )
    if method == 'policy':
        uci = predict_next_move(
            model, board,
            temperature=args.temperature,
            value_weight=args.value_weight,
            value_candidate_limit=args.value_candidates,
        )
        return chess.Move.from_uci(uci) if uci else None
    if method == 'search':
        return search_best_move(
            model, board, max_depth=args.search_depth, time_limit=args.search_time,
        )
    raise ValueError(f"Unknown method: {method}")


def play_one_game(model_white, method_white, model_black, method_black,
                  args) -> str:
    """Play a single game. Returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= args.max_plies:
            return '1/2-1/2'  # adjudicate overlong games as draws
        if board.turn == chess.WHITE:
            move = pick_move(method_white, model_white, board, args)
        else:
            move = pick_move(method_black, model_black, board, args)
        if move is None or move not in board.legal_moves:
            # A model that fails to produce a legal move forfeits.
            return '0-1' if board.turn == chess.WHITE else '1-0'
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    return outcome.result() if outcome else '1/2-1/2'


def elo_diff_and_ci(scores: list[float]) -> tuple[float, float, float, float]:
    """From per-game scores in {0, 0.5, 1} (A's POV), return
    (mean_score, elo_diff, elo_low, elo_high) with a 95% CI."""
    n = len(scores)
    s = float(np.mean(scores)) if n else 0.5

    def to_elo(score: float) -> float:
        score = min(max(score, 1e-6), 1 - 1e-6)
        return 400.0 * math.log10(score / (1.0 - score))

    # Standard error of the mean score; map the score CI through the Elo curve.
    if n > 1:
        se = float(np.std(scores, ddof=1)) / math.sqrt(n)
    else:
        se = 0.0
    lo = min(max(s - 1.96 * se, 1e-6), 1 - 1e-6)
    hi = min(max(s + 1.96 * se, 1e-6), 1 - 1e-6)
    return s, to_elo(s), to_elo(lo), to_elo(hi)


def run_match(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model A: {args.model_a}")
    model_a = load_trained_model(args.model_a)
    if args.model_b == args.model_a:
        print("Model B is the same checkpoint as A (sanity / self-match).")
        model_b = model_a
    else:
        print(f"Loading model B: {args.model_b}")
        model_b = load_trained_model(args.model_b)

    method_a, method_b = args.method_a, args.method_b
    results = {'A_win': 0, 'B_win': 0, 'draw': 0}
    a_scores: list[float] = []  # A's score per game

    for game_idx in range(1, args.games + 1):
        a_is_white = (game_idx % 2 == 1)
        if a_is_white:
            res = play_one_game(model_a, method_a, model_b, method_b, args)
        else:
            res = play_one_game(model_b, method_b, model_a, method_a, args)

        # Translate board result into A's perspective.
        if res == '1/2-1/2':
            a_score = 0.5
            results['draw'] += 1
        else:
            white_won = (res == '1-0')
            a_won = (white_won == a_is_white)
            a_score = 1.0 if a_won else 0.0
            results['A_win' if a_won else 'B_win'] += 1
        a_scores.append(a_score)

        running = float(np.mean(a_scores))
        print(
            f"  Game {game_idx}/{args.games}: A={'W' if a_is_white else 'B'} "
            f"result={res} -> A_score={a_score} "
            f"(running A={running:.3f}, +{results['A_win']}={results['draw']}"
            f"-{results['B_win']})"
        )

    mean_score, elo, elo_lo, elo_hi = elo_diff_and_ci(a_scores)
    print()
    print("=" * 56)
    print(f"A: {args.model_a}  [{method_a}]")
    print(f"B: {args.model_b}  [{method_b}]")
    print(f"Games: {args.games}  |  A wins={results['A_win']} "
          f"draws={results['draw']} losses={results['B_win']}")
    print(f"A score: {mean_score:.3f}")
    print(f"Elo(A - B): {elo:+.0f}  (95% CI [{elo_lo:+.0f}, {elo_hi:+.0f}])")
    if elo_lo > 0:
        print("=> A is stronger than B with 95% confidence.")
    elif elo_hi < 0:
        print("=> B is stronger than A with 95% confidence.")
    else:
        print("=> Inconclusive at 95% confidence; play more games.")
    print("=" * 56)
    return {'results': results, 'mean_score': mean_score, 'elo': elo,
            'elo_ci': (elo_lo, elo_hi)}


def main():
    parser = argparse.ArgumentParser(
        description='Head-to-head match between two checkpoints with Elo estimate.'
    )
    parser.add_argument('--model_a', required=True, help='Checkpoint A (e.g. a self-play iteration).')
    parser.add_argument('--model_b', required=True, help='Checkpoint B (e.g. the supervised base).')
    parser.add_argument('--method_a', choices=('mcts', 'policy', 'search'), default='mcts')
    parser.add_argument('--method_b', choices=('mcts', 'policy', 'search'), default='mcts')
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--max_plies', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    # MCTS params
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--c_puct', type=float, default=1.5)
    parser.add_argument('--mcts_batch_size', type=int, default=16)
    parser.add_argument('--policy_temperature', type=float, default=1.5)
    parser.add_argument('--move_temperature', type=float, default=0.0)
    # Policy params
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--value_weight', type=float, default=2.0)
    parser.add_argument('--value_candidates', type=int, default=0)
    # Search params
    parser.add_argument('--search_depth', type=int, default=3)
    parser.add_argument('--search_time', type=float, default=None)
    args = parser.parse_args()
    run_match(args)


if __name__ == '__main__':
    main()
