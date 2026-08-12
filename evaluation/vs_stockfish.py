"""Reproducible model-vs-Stockfish evaluation.

The default protocol plays every built-in opening twice with colors reversed.
It records the exact configuration, artifact identities, PGNs, per-model-move
latency, and confidence intervals.  A UCI_LimitStrength result is deliberately
reported as a result against that Stockfish setting under this harness; it is
not presented as a human-rating estimate.

Example:
  python -m evaluation.vs_stockfish \
      --uci_elo 2500 --mode mcts --sims 200 --output_dir evaluation/results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from evaluation.eval_arena import (
    DEFAULT_OPENINGS,
    _board_from_opening,
    elo_diff_and_ci,
)
from inference.mcts_player import mcts_search_best_move
from load_model import load_trained_model, predict_next_move
from neural_network import MODEL_PATH


def _sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _score_ci(scores: list[float]) -> tuple[float, float]:
    """Normal 95% interval over independent evaluation units.

    For paired mode each unit is the mean of the two color-reversed games for
    an opening, so color-correlated games are not counted as independent.
    """
    if len(scores) < 2:
        value = scores[0] if scores else 0.5
        return value, value
    mean = statistics.fmean(scores)
    se = statistics.stdev(scores) / math.sqrt(len(scores))
    return max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)


def _read_openings(path: str | None, no_openings: bool) -> list[tuple[str, str]]:
    if no_openings:
        return [('startpos', '')]
    if path is None:
        return list(DEFAULT_OPENINGS.items())
    openings = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            moves = line.strip()
            if moves and not moves.startswith('#'):
                openings.append((f'line-{line_number}', moves))
    if not openings:
        raise SystemExit(f'No openings found in {path!r}.')
    return openings


def _schedule(args) -> list[tuple[str, str, bool, int]]:
    openings = _read_openings(args.openings, args.no_openings)
    target = args.games or (2 * len(openings) if args.paired else len(openings))
    if args.paired and target % 2:
        raise SystemExit('--games must be even when paired openings are enabled.')
    schedule: list[tuple[str, str, bool, int]] = []
    cycle = 0
    while len(schedule) < target:
        ordered = list(openings)
        if cycle:
            random.Random(args.seed + cycle).shuffle(ordered)
        for name, moves in ordered:
            if args.paired:
                schedule.extend([
                    (name, moves, True, cycle),
                    (name, moves, False, cycle),
                ])
            else:
                schedule.append(
                    (name, moves, len(schedule) % 2 == 0, cycle)
                )
            if len(schedule) >= target:
                break
        cycle += 1
    return schedule[:target]


def model_move(model, board: chess.Board, args) -> chess.Move | None:
    if args.mode == 'policy':
        uci = predict_next_move(
            model,
            board,
            temperature=args.temperature,
            value_weight=args.value_weight,
            value_candidate_limit=args.value_candidates,
            blunder_guard=not args.no_blunder_guard,
            blunder_guard_depth=args.guard_depth,
        )
        return chess.Move.from_uci(uci) if uci else None
    return mcts_search_best_move(
        model,
        board,
        num_simulations=args.sims,
        batch_size=args.mcts_batch_size,
        policy_temperature=args.mcts_policy_temp,
        move_temperature=0.0,
        time_limit=(args.model_time_limit or None),
    )


def play_game(model, engine, args, opening_name: str, opening_moves: str,
              model_is_white: bool, round_number: int) -> dict:
    board = _board_from_opening(opening_moves)
    if board is None:
        raise ValueError(f'Illegal opening {opening_name!r}: {opening_moves}')
    game = chess.pgn.Game()
    game.headers.update({
        'Event': 'playable-chess-AI reproducible evaluation',
        'Site': 'local',
        'Date': datetime.now(timezone.utc).strftime('%Y.%m.%d'),
        'Round': str(round_number),
        'White': 'model' if model_is_white else 'Stockfish',
        'Black': 'Stockfish' if model_is_white else 'model',
        'Opening': opening_name,
    })
    # Keep the prescribed opening in the PGN rather than treating the
    # post-opening board as an opaque FEN. This makes every game independently
    # replayable and lets reviewers audit the color-reversed pair.
    node = game
    for opening_move in board.move_stack:
        node = node.add_variation(opening_move)
    model_latencies: list[float] = []
    illegal = None

    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= args.max_plies:
            result = '1/2-1/2'
            break
        if (board.turn == chess.WHITE) == model_is_white:
            started = time.perf_counter()
            move = model_move(model, board, args)
            model_latencies.append(time.perf_counter() - started)
        else:
            move = engine.play(
                board, chess.engine.Limit(time=args.movetime)
            ).move
        if move is None or move not in board.legal_moves:
            illegal = None if move is None else move.uci()
            result = '0-1' if board.turn == chess.WHITE else '1-0'
            break
        board.push(move)
        node = node.add_variation(move)
    else:
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else '1/2-1/2'

    game.headers['Result'] = result
    model_score = (
        0.5 if result == '1/2-1/2'
        else float((result == '1-0') == model_is_white)
    )
    return {
        'opening': opening_name,
        'opening_moves': opening_moves,
        'model_color': 'white' if model_is_white else 'black',
        'result': result,
        'model_score': model_score,
        'plies': len(board.move_stack),
        'model_move_seconds': model_latencies,
        'illegal_or_missing_model_move': illegal,
        'pgn': str(game),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--mode', choices=('policy', 'mcts'), default='policy')
    parser.add_argument('--games', type=int, default=0,
                        help='0 uses the full opening suite (32 paired games).')
    parser.add_argument('--seed', type=int, default=20260705)
    parser.add_argument('--openings', default=None,
                        help='File with one space-separated UCI line per opening.')
    parser.add_argument('--no_openings', action='store_true',
                        help='Legacy start-position-only protocol.')
    parser.add_argument('--paired', action='store_true', default=True)
    parser.add_argument('--no_paired', dest='paired', action='store_false')
    parser.add_argument('--skill', type=int, default=10)
    parser.add_argument('--uci_elo', type=int, default=None)
    parser.add_argument('--movetime', type=float, default=0.6)
    parser.add_argument('--max_plies', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--value_weight', type=float, default=2.0)
    parser.add_argument('--value_candidates', type=int, default=0)
    parser.add_argument('--no_blunder_guard', action='store_true')
    parser.add_argument('--guard_depth', type=int, default=2)
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--mcts_batch_size', type=int, default=16)
    parser.add_argument('--mcts_policy_temp', type=float, default=1.5)
    parser.add_argument('--model_time_limit', type=float, default=0.0,
                        help='Soft MCTS deadline in seconds; 0 disables it.')
    parser.add_argument('--output_dir', default='evaluation/results')
    parser.add_argument('--require_score_lower_bound', type=float, default=None,
                        help='Exit nonzero unless the 95%% lower score bound meets this.')
    parser.add_argument('--require_p95_seconds', type=float, default=None,
                        help='Exit nonzero unless model move p95 is at most this.')
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        args.games < 0 or args.movetime <= 0 or args.max_plies < 1 or
        args.seed < 0 or args.sims < 1 or args.mcts_batch_size < 1 or
        args.model_time_limit < 0
    ):
        raise SystemExit(
            'require games>=0, seed>=0, movetime>0, max_plies>=1, sims>=1, '
            'mcts_batch_size>=1, and model_time_limit>=0.'
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_trained_model(args.model)
    engine_path = os.environ.get('STOCKFISH_PATH', 'stockfish.exe')
    if not os.path.isabs(engine_path) and os.path.exists(engine_path):
        engine_path = os.path.abspath(engine_path)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine_identity = dict(engine.id)
    if args.uci_elo is not None:
        elo = max(1320, min(args.uci_elo, 3190))
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': elo})
        opponent = f'Stockfish UCI_LimitStrength UCI_Elo={elo}'
    else:
        engine.configure({'Skill Level': max(0, min(args.skill, 20))})
        opponent = f'Stockfish Skill Level={max(0, min(args.skill, 20))}'

    schedule = _schedule(args)
    print(f'{args.mode} vs {opponent}; {len(schedule)} games; '
          f'{"paired openings" if args.paired else "unpaired"}', flush=True)
    games = []
    try:
        for index, (name, moves, model_white, cycle) in enumerate(schedule, 1):
            game_seed = args.seed + 1000 * cycle + index
            random.seed(game_seed)
            np.random.seed(game_seed % (2**32 - 1))
            torch.manual_seed(game_seed)
            started = time.perf_counter()
            game = play_game(
                model, engine, args, name, moves, model_white, index
            )
            game['elapsed_seconds'] = time.perf_counter() - started
            game['seed'] = game_seed
            games.append(game)
            wins = sum(g['model_score'] == 1 for g in games)
            draws = sum(g['model_score'] == 0.5 for g in games)
            losses = sum(g['model_score'] == 0 for g in games)
            print(f'  {index:02d}/{len(schedule)} {name} model '
                  f'{game["model_color"]}: {game["result"]} '
                  f'[{wins}W-{draws}D-{losses}L]', flush=True)
    finally:
        engine.quit()

    scores = [float(game['model_score']) for game in games]
    paired_units = (
        [statistics.fmean(scores[i:i + 2]) for i in range(0, len(scores), 2)]
        if args.paired else scores
    )
    mean_score, elo_delta, elo_low, elo_high = elo_diff_and_ci(paired_units)
    score_low, score_high = _score_ci(paired_units)
    latencies = [
        latency for game in games for latency in game['model_move_seconds']
    ]
    latency = {
        'count': len(latencies),
        'mean': statistics.fmean(latencies) if latencies else None,
        'p50': _percentile(latencies, 0.50),
        'p95': _percentile(latencies, 0.95),
        'max': max(latencies) if latencies else None,
    }
    wins = sum(score == 1 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0 for score in scores)

    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S.%fZ')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = output_dir / f'vs_stockfish_{run_id}.pgn'
    json_path = output_dir / f'vs_stockfish_{run_id}.json'
    pgn_path.write_text('\n\n'.join(game['pgn'] for game in games) + '\n',
                        encoding='utf-8')

    manifest = {
        'schema_version': 1,
        'created_at': now.isoformat(),
        'command': [sys.executable, '-m', 'evaluation.vs_stockfish', *sys.argv[1:]],
        'protocol': {
            'paired_openings': args.paired,
            'startpos_only': args.no_openings,
            'seed': args.seed,
            'requested_games': args.games,
            'played_games': len(games),
            'independent_units_for_ci': len(paired_units),
            'opponent': opponent,
            'stockfish_movetime_seconds': args.movetime,
            'mode': args.mode,
            'model_soft_time_limit_seconds': args.model_time_limit or None,
            'arguments': vars(args),
        },
        'artifacts': {
            'code_git_sha': _git_sha(),
            'model_path': os.path.abspath(getattr(model, 'checkpoint_path', args.model)),
            'model_sha256': getattr(model, 'checkpoint_sha256', _sha256(args.model)),
            'model_architecture': getattr(model, 'arch_version', None),
            'model_encoding': getattr(model, 'encoding_version', None),
            'model_hf_repo': getattr(model, 'hf_repo', None),
            'model_hf_revision': getattr(model, 'hf_revision', None),
            'stockfish_path': os.path.abspath(engine_path),
            'stockfish_sha256': _sha256(engine_path),
            'stockfish_id': engine_identity,
            'openings_path': os.path.abspath(args.openings) if args.openings else None,
            'openings_sha256': _sha256(args.openings) if args.openings else None,
            'builtin_openings': None if args.openings else DEFAULT_OPENINGS,
            'pgn': str(pgn_path),
        },
        'environment': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'torch': str(torch.__version__),
            'device': str(next(model.parameters()).device),
            'device_name': (
                torch.cuda.get_device_name(next(model.parameters()).device)
                if next(model.parameters()).device.type == 'cuda' else None
            ),
            'torch_threads': torch.get_num_threads(),
        },
        'summary': {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'score': statistics.fmean(scores) if scores else 0.5,
            'paired_unit_score': mean_score,
            'score_95_ci': [score_low, score_high],
            'elo_delta_vs_configured_stockfish': elo_delta,
            'elo_delta_95_ci': [elo_low, elo_high],
            'model_move_latency_seconds': latency,
        },
        'games': games,
    }
    json_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    print(f'\n{wins}W-{draws}D-{losses}L; raw score '
          f'{statistics.fmean(scores):.1%}; paired-unit score {mean_score:.1%} '
          f'(95% CI {score_low:.1%}-{score_high:.1%})')
    print(f'Elo delta vs configured Stockfish under this harness: '
          f'{elo_delta:+.0f} (95% CI {elo_low:+.0f} to {elo_high:+.0f})')
    print(f'Model latency p50={latency["p50"]:.3f}s '
          f'p95={latency["p95"]:.3f}s max={latency["max"]:.3f}s')
    print(f'Artifacts: {json_path} and {pgn_path}')

    failures = []
    if (args.require_score_lower_bound is not None and
            score_low < args.require_score_lower_bound):
        failures.append(
            f'score lower bound {score_low:.3f} < '
            f'{args.require_score_lower_bound:.3f}'
        )
    if (args.require_p95_seconds is not None and
            (latency['p95'] is None or latency['p95'] > args.require_p95_seconds)):
        failures.append(
            f'model p95 {latency["p95"]} > {args.require_p95_seconds:.3f}s'
        )
    if failures:
        raise SystemExit('Evaluation gate failed: ' + '; '.join(failures))


if __name__ == '__main__':
    main()
