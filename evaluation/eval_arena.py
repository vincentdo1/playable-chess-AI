"""Head-to-head evaluation between two checkpoints, with an Elo estimate.

Upgrades over the original single-startpos arena:
  * Opening suite: games start from a set of diverse opening lines instead of
    only the initial position, so a strength edge in one structure does not
    dominate the result. A built-in suite of common openings is the default;
    pass --openings <file> with one opening per line (UCI moves separated by
    spaces, e.g. "e2e4 e7e5 g1f3") to supply your own.
  * Paired games: each opening is played TWICE with colors swapped, and both
    games count. This cancels first-move/color advantage and opening-side
    advantage at the level of every pair, which is the cleanest way to make a
    20-game match actually informative when play is near-deterministic.
  * Same Elo + 95% CI reporting as before; --no_openings reproduces the old
    start-from-initial behavior for backward compatibility.

Methods per side: 'mcts' (search), 'policy' (single forward pass, temperature
sampling), or 'search' (alpha-beta with the network as evaluator).
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


# A compact, structurally-diverse opening suite (UCI move lists). These are
# only a few plies deep so models still make most of the game's decisions, but
# deep enough to vary pawn structure and piece deployment across games.
DEFAULT_OPENINGS = {
    'startpos': '',
    'open_e4e5': 'e2e4 e7e5',
    'ruy_lopez': 'e2e4 e7e5 g1f3 b8c6 f1b5',
    'italian': 'e2e4 e7e5 g1f3 b8c6 f1c4',
    'sicilian': 'e2e4 c7c5',
    'sicilian_najdorf': 'e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6',
    'french': 'e2e4 e7e6 d2d4 d7d5',
    'caro_kann': 'e2e4 c7c6 d2d4 d7d5',
    'queens_gambit': 'd2d4 d7d5 c2c4',
    'qgd': 'd2d4 d7d5 c2c4 e7e6',
    'slav': 'd2d4 d7d5 c2c4 c7c6',
    'kings_indian': 'd2d4 g8f6 c2c4 g7g6 b1c3 f8g7',
    'nimzo_indian': 'd2d4 g8f6 c2c4 e7e6 b1c3 f8b4',
    'english': 'c2c4 e7e5',
    'london': 'd2d4 d7d5 g1f3 g8f6 c1f4',
    'reti': 'g1f3 d7d5 c2c4',
}


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


def _board_from_opening(opening_moves: str) -> chess.Board | None:
    """Build a board from a space-separated UCI opening; None if illegal."""
    board = chess.Board()
    for uci in opening_moves.split():
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
    return board


def play_one_game(model_white, method_white, model_black, method_black,
                  args, start_board: chess.Board) -> str:
    """Play a single game from start_board. Returns '1-0', '0-1', or '1/2-1/2'."""
    board = start_board.copy()
    if board.is_game_over(claim_draw=True):
        return '1/2-1/2'
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= args.max_plies:
            return '1/2-1/2'  # adjudicate overlong games as draws
        if board.turn == chess.WHITE:
            move = pick_move(method_white, model_white, board, args)
        else:
            move = pick_move(method_black, model_black, board, args)
        if move is None or move not in board.legal_moves:
            return '0-1' if board.turn == chess.WHITE else '1-0'
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    return outcome.result() if outcome else '1/2-1/2'


def _a_score_from_result(res: str, a_is_white: bool) -> float:
    if res == '1/2-1/2':
        return 0.5
    return 1.0 if ((res == '1-0') == a_is_white) else 0.0


def elo_diff_and_ci(scores: list[float]) -> tuple[float, float, float, float]:
    """From per-game scores in {0, 0.5, 1} (A's POV), return
    (mean_score, elo_diff, elo_low, elo_high) with a 95% CI."""
    n = len(scores)
    s = float(np.mean(scores)) if n else 0.5

    def to_elo(score: float) -> float:
        score = min(max(score, 1e-6), 1 - 1e-6)
        return 400.0 * math.log10(score / (1.0 - score))

    if n > 1:
        se = float(np.std(scores, ddof=1)) / math.sqrt(n)
    else:
        se = 0.0
    lo = min(max(s - 1.96 * se, 1e-6), 1 - 1e-6)
    hi = min(max(s + 1.96 * se, 1e-6), 1 - 1e-6)
    return s, to_elo(s), to_elo(lo), to_elo(hi)


def _load_openings(args) -> list[tuple[str, str]]:
    if args.no_openings:
        return [('startpos', '')]
    if args.openings:
        pairs = []
        with open(args.openings, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('#'):
                    pairs.append((f'line{i}', line))
        if not pairs:
            raise SystemExit(f"No openings found in {args.openings}")
        return pairs
    return list(DEFAULT_OPENINGS.items())


def run_match(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model A: {args.model_a}", flush=True)
    model_a = load_trained_model(args.model_a)
    if args.model_b == args.model_a:
        print("Model B is the same checkpoint as A (sanity / self-match).", flush=True)
        model_b = model_a
    else:
        print(f"Loading model B: {args.model_b}", flush=True)
        model_b = load_trained_model(args.model_b)

    method_a, method_b = args.method_a, args.method_b
    openings = _load_openings(args)

    # Build the game schedule. With paired play, each opening contributes two
    # games (A-white then A-black). When --games exceeds one pass over the
    # suite, we repeat the paired schedule in additional CYCLES; each cycle
    # reshuffles opening order and bumps the RNG seed so the repeated games
    # actually diverge (otherwise deterministic play would replay identical
    # games and falsely shrink the confidence interval).
    schedule: list[tuple[str, str, bool, int]] = []  # (name, moves, a_is_white, seed)
    if args.paired:
        base_pairs = [(name, moves) for name, moves in openings]
        full_pass = 2 * len(base_pairs)
        target = args.games if args.games > 0 else full_pass
        cycle = 0
        while len(schedule) < target:
            order = list(base_pairs)
            if cycle > 0:
                # Reshuffle order and vary seed on every cycle past the first.
                rng = random.Random(args.seed + cycle)
                rng.shuffle(order)
            for name, moves in order:
                cycle_seed = args.seed + cycle
                schedule.append((name, moves, True, cycle_seed))
                schedule.append((name, moves, False, cycle_seed))
                if len(schedule) >= target:
                    break
            cycle += 1
        schedule = schedule[:target]
    else:
        idx = 0
        target = args.games if args.games > 0 else len(openings)
        while len(schedule) < target:
            name, moves = openings[idx % len(openings)]
            schedule.append((name, moves, len(schedule) % 2 == 0, args.seed + idx))
            idx += 1

    results = {'A_win': 0, 'B_win': 0, 'draw': 0}
    a_scores: list[float] = []
    skipped = 0

    for gi, (name, moves, a_is_white, game_seed) in enumerate(schedule, start=1):
        start_board = _board_from_opening(moves)
        if start_board is None:
            print(f"  [skip] illegal opening {name!r}: {moves}", flush=True)
            skipped += 1
            continue
        # Re-seed per game so repeated cycles of the same opening diverge but the
        # whole match stays reproducible for a fixed --seed.
        random.seed(game_seed + gi)
        np.random.seed((game_seed + gi) % (2**32 - 1))
        if a_is_white:
            res = play_one_game(model_a, method_a, model_b, method_b, args, start_board)
        else:
            res = play_one_game(model_b, method_b, model_a, method_a, args, start_board)

        a_score = _a_score_from_result(res, a_is_white)
        if a_score == 0.5:
            results['draw'] += 1
        elif a_score == 1.0:
            results['A_win'] += 1
        else:
            results['B_win'] += 1
        a_scores.append(a_score)

        running = float(np.mean(a_scores))
        print(
            f"  Game {gi}/{len(schedule)} [{name}]: A={'W' if a_is_white else 'B'} "
            f"result={res} -> A={a_score} "
            f"(running A={running:.3f}, +{results['A_win']}={results['draw']}"
            f"-{results['B_win']})",
            flush=True,
        )

    mean_score, elo, elo_lo, elo_hi = elo_diff_and_ci(a_scores)
    print()
    print("=" * 60)
    print(f"A: {args.model_a}  [{method_a}]")
    print(f"B: {args.model_b}  [{method_b}]")
    mode = 'paired openings' if args.paired else (
        'opening suite' if not args.no_openings else 'startpos only')
    print(f"Mode: {mode}  |  games played: {len(a_scores)}"
          + (f"  (skipped {skipped})" if skipped else ""))
    print(f"A wins={results['A_win']} draws={results['draw']} losses={results['B_win']}")
    print(f"A score: {mean_score:.3f}")
    print(f"Elo(A - B): {elo:+.0f}  (95% CI [{elo_lo:+.0f}, {elo_hi:+.0f}])")
    if elo_lo > 0:
        print("=> A is stronger than B with 95% confidence.")
    elif elo_hi < 0:
        print("=> B is stronger than A with 95% confidence.")
    else:
        print("=> Inconclusive at 95% confidence; play more games.")
    print("=" * 60, flush=True)
    return {'results': results, 'mean_score': mean_score, 'elo': elo,
            'elo_ci': (elo_lo, elo_hi), 'games': len(a_scores)}


def main():
    parser = argparse.ArgumentParser(
        description='Head-to-head match between two checkpoints with Elo estimate.'
    )
    parser.add_argument('--model_a', required=True)
    parser.add_argument('--model_b', required=True)
    parser.add_argument('--method_a', choices=('mcts', 'policy', 'search'), default='mcts')
    parser.add_argument('--method_b', choices=('mcts', 'policy', 'search'), default='mcts')
    # Opening / pairing controls
    parser.add_argument('--paired', action='store_true', default=True,
                        help='Play each opening twice with swapped colors (default on).')
    parser.add_argument('--no_paired', dest='paired', action='store_false',
                        help='Disable paired play; one game per opening slot.')
    parser.add_argument('--openings', default=None,
                        help='File with one opening per line (space-separated UCI). '
                             'Defaults to a built-in diverse suite.')
    parser.add_argument('--no_openings', action='store_true',
                        help='Start every game from the initial position (old behavior).')
    parser.add_argument('--games', type=int, default=0,
                        help='Cap on games. 0 = use the full suite (x2 if --paired).')
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
