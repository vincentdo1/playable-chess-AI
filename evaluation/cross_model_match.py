"""Play a match between two models that may live in different code branches.

Each side runs in its own subprocess (move_server.py) with its own working
directory and checkpoint, so they can use incompatible architectures or
incompatible versions of load_model.py / neural_network.py and still play
each other through a JSON line protocol.

Typical use: pit the older repo checkpoint (under a main-branch worktree)
against your freshly trained current-architecture checkpoint (this branch).

Usage example, run from this branch:
  # 1. Create a worktree of main alongside your repo:
  #      git worktree add ../playable-chess-ai-main main
  #      copy any needed checkpoints into ../playable-chess-ai-main/model/
  #      copy move_server.py into ../playable-chess-ai-main/
  # 2. Pit them:
  python cross_model_match.py \
    --player_a_dir ../playable-chess-ai-main \
    --player_a_model model/grandmaster_model_policy_v1.pt \
    --player_a_method policy \
    --player_b_dir . \
    --player_b_model model/grandmaster_model_perspective_resnet_negatives_v2.pt \
    --player_b_method mcts --player_b_sims 200 \
    --games 10 --output cross_match.pgn

The driver alternates colors so neither model gets a first-move advantage,
reports a per-game score and a final Elo gap with a 95% CI, and optionally
writes the games to a PGN you can view on lichess.org/paste or any GUI.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime

import chess
import chess.pgn
import numpy as np


def launch(working_dir: str, python_exe: str, model_path: str | None):
    cmd = [python_exe, 'move_server.py']
    if model_path:
        cmd += ['--model_path', model_path]
    proc = subprocess.Popen(
        cmd,
        cwd=working_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        raise RuntimeError(f"player at {working_dir!r} failed to start: {err}")
    ack = json.loads(line.strip())
    if not ack.get('ready'):
        raise RuntimeError(f"player at {working_dir!r} not ready: {ack}")
    print(
        f"  launched player from {working_dir}: model={ack.get('model_path')} "
        f"has_mcts={ack.get('has_mcts')}",
        flush=True,
    )
    return proc


def get_move(proc: subprocess.Popen, history: list[str], **kwargs) -> str | None:
    request = {'history': history, **kwargs}
    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"player died mid-game: {proc.stderr.read()}")
    response = json.loads(line.strip())
    if 'error' in response:
        raise RuntimeError(f"player error: {response['error']}")
    return response.get('move')


def play_game(player_a: subprocess.Popen, player_b: subprocess.Popen,
              a_is_white: bool, a_kwargs: dict, b_kwargs: dict,
              max_plies: int, game_seed: int) -> tuple[str, chess.Board]:
    board = chess.Board()
    history: list[str] = []
    # Seed each player once, on its first move this game, so sampled play
    # diverges between games while staying reproducible for a fixed --seed.
    seeded = {id(player_a): False, id(player_b): False}
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= max_plies:
            return '1/2-1/2', board
        a_to_move = (board.turn == chess.WHITE) == a_is_white
        proc = player_a if a_to_move else player_b
        kwargs = dict(a_kwargs if a_to_move else b_kwargs)
        if not seeded[id(proc)]:
            # Offset A vs B so the two players don't share an identical stream.
            kwargs['seed'] = game_seed + (0 if proc is player_a else 100000)
            seeded[id(proc)] = True
        try:
            uci = get_move(proc, history, **kwargs)
        except RuntimeError as exc:
            print(f"  ! {exc}", flush=True)
            return ('0-1' if board.turn == chess.WHITE else '1-0'), board
        if not uci:
            return ('0-1' if board.turn == chess.WHITE else '1-0'), board
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return ('0-1' if board.turn == chess.WHITE else '1-0'), board
        if move not in board.legal_moves:
            return ('0-1' if board.turn == chess.WHITE else '1-0'), board
        board.push(move)
        history.append(uci)
    outcome = board.outcome(claim_draw=True)
    return (outcome.result() if outcome else '1/2-1/2'), board


def elo_diff_and_ci(scores: list[float]) -> tuple[float, float, float, float]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--player_a_dir', required=True)
    parser.add_argument('--player_a_model', default=None)
    parser.add_argument('--player_a_method', choices=('policy', 'mcts'), default='policy')
    parser.add_argument('--player_a_sims', type=int, default=200)
    parser.add_argument('--player_a_temperature', type=float, default=0.0,
                        help='Policy sampling temperature for A. >0 makes games '
                             'diverge so the Elo/CI are statistically meaningful.')
    parser.add_argument('--player_a_python', default=sys.executable)
    parser.add_argument('--player_b_dir', required=True)
    parser.add_argument('--player_b_model', default=None)
    parser.add_argument('--player_b_method', choices=('policy', 'mcts'), default='policy')
    parser.add_argument('--player_b_sims', type=int, default=200)
    parser.add_argument('--player_b_temperature', type=float, default=0.0,
                        help='Policy sampling temperature for B.')
    parser.add_argument('--player_b_python', default=sys.executable)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--max_plies', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1,
                        help='Base seed; game i uses seed+i so games diverge but '
                             'the whole match is reproducible.')
    parser.add_argument('--output', default=None, help='Write all games to this PGN path.')
    args = parser.parse_args()

    def kwargs_for(method, sims, temperature):
        if method == 'mcts':
            # move_temperature drives MCTS root-move sampling; reuse the same
            # knob so >0 yields game variety for the search player too.
            return {'method': method, 'sims': sims, 'move_temperature': temperature}
        return {'method': method, 'temperature': temperature}

    a_kwargs = kwargs_for(args.player_a_method, args.player_a_sims, args.player_a_temperature)
    b_kwargs = kwargs_for(args.player_b_method, args.player_b_sims, args.player_b_temperature)

    print(f"Launching player A from {args.player_a_dir}", flush=True)
    player_a = launch(args.player_a_dir, args.player_a_python, args.player_a_model)
    print(f"Launching player B from {args.player_b_dir}", flush=True)
    player_b = launch(args.player_b_dir, args.player_b_python, args.player_b_model)

    pgn_games: list[chess.pgn.Game] = []
    a_scores: list[float] = []
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}

    try:
        for i in range(args.games):
            a_is_white = (i % 2 == 0)
            print(f"Game {i+1}/{args.games}: A={'W' if a_is_white else 'B'}", flush=True)
            result, final_board = play_game(
                player_a, player_b, a_is_white, a_kwargs, b_kwargs, args.max_plies,
                game_seed=args.seed + i,
            )

            if result == '1/2-1/2':
                a_scores.append(0.5)
            elif (result == '1-0') == a_is_white:
                a_scores.append(1.0)
            else:
                a_scores.append(0.0)
            results[result] = results.get(result, 0) + 1

            print(
                f"  -> {result} (A_score={a_scores[-1]}, "
                f"running={np.mean(a_scores):.3f})",
                flush=True,
            )

            if args.output:
                game = chess.pgn.Game()
                game.headers['Event'] = 'Cross-model match'
                game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')
                game.headers['Round'] = str(i + 1)
                game.headers['White'] = (
                    f"A:{args.player_a_model or 'default'}" if a_is_white
                    else f"B:{args.player_b_model or 'default'}"
                )
                game.headers['Black'] = (
                    f"B:{args.player_b_model or 'default'}" if a_is_white
                    else f"A:{args.player_a_model or 'default'}"
                )
                game.headers['Result'] = result
                node = game
                for move in final_board.move_stack:
                    node = node.add_variation(move)
                pgn_games.append(game)
    finally:
        for proc in (player_a, player_b):
            try:
                proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                proc.stdin.flush()
            except Exception:
                pass
            proc.terminate()

    if args.output and pgn_games:
        with open(args.output, 'w', encoding='utf-8') as f:
            for game in pgn_games:
                print(game, file=f, end='\n\n')
        print(f"Wrote PGN: {args.output}", flush=True)

    mean_score, elo, elo_lo, elo_hi = elo_diff_and_ci(a_scores)
    print()
    print('=' * 56)
    print(f"A: {args.player_a_model} @ {args.player_a_dir}  [{args.player_a_method}]")
    print(f"B: {args.player_b_model} @ {args.player_b_dir}  [{args.player_b_method}]")
    print(f"Games: {args.games}  |  +{results['1-0']}={results['1/2-1/2']}-{results['0-1']}")
    print(f"A score: {mean_score:.3f}")
    print(f"Elo(A - B): {elo:+.0f}  (95% CI [{elo_lo:+.0f}, {elo_hi:+.0f}])")
    print('=' * 56)
    return 0


if __name__ == '__main__':
    sys.exit(main())
