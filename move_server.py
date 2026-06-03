"""Stdin/stdout move server for cross-branch model evaluation.

Lets a model running in one Python process/working-dir be driven by another.
The companion driver, cross_model_match.py, spawns one of these per player
and pipes requests/responses over JSON line protocol.

Why this exists: comparing models trained against different architectures
(e.g. the older repo checkpoint vs. a freshly trained current-architecture
checkpoint) means each must run in its own branch's code. Subprocess
isolation keeps each model in its native environment.

Protocol:
  Startup ack    : server writes one JSON line, e.g.
                   {"ready": true, "model_path": "...", "has_mcts": true}
  Request        : driver writes one JSON line per move:
                   {"history": ["e2e4","e7e5"], "method": "policy"|"mcts", ...}
  Response       : server writes one JSON line:
                   {"move": "g1f3"}   or   {"error": "..."}
  Shutdown       : driver sends {"cmd":"quit"} or closes stdin.

Works on any branch that has load_model.load_trained_model + predict_next_move.
MCTS method also requires mcts_player.mcts_search_best_move; if absent, the
server reports has_mcts=false at startup and rejects mcts requests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback

import chess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
                        help='Checkpoint to load; defaults to the branch MODEL_PATH.')
    args = parser.parse_args()

    # load_trained_model prints status lines to stdout; route any such prints
    # to stderr so the JSON line protocol on stdout stays clean.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from load_model import load_trained_model, predict_next_move
            model = (
                load_trained_model(args.model_path)
                if args.model_path else load_trained_model()
            )
    except Exception as exc:
        sys.stdout.write(json.dumps({
            'ready': False, 'error': str(exc),
        }) + '\n')
        sys.stdout.flush()
        return 1

    mcts_fn = None
    try:
        from mcts_player import mcts_search_best_move
        mcts_fn = mcts_search_best_move
    except ImportError:
        pass

    sys.stdout.write(json.dumps({
        'ready': True,
        'model_path': args.model_path,
        'has_mcts': mcts_fn is not None,
    }) + '\n')
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get('cmd') == 'quit':
                break

            board = chess.Board()
            for uci in req.get('history', []):
                board.push(chess.Move.from_uci(uci))

            # Optional per-request seed so the driver can make each game's
            # sampled moves diverge (otherwise temperature>0 still repeats
            # because numpy's global RNG state is process-wide).
            seed = req.get('seed')
            if seed is not None:
                import numpy as _np
                import random as _random
                _np.random.seed(int(seed))
                _random.seed(int(seed))

            method = req.get('method', 'policy')
            if method == 'mcts':
                if mcts_fn is None:
                    raise RuntimeError(
                        'mcts requested but mcts_player is not importable '
                        'in this branch.'
                    )
                move = mcts_fn(
                    model, board,
                    num_simulations=int(req.get('sims', 200)),
                    batch_size=int(req.get('batch_size', 16)),
                    policy_temperature=float(req.get('policy_temperature', 1.5)),
                    move_temperature=float(req.get('move_temperature', 0.0)),
                )
                response = {'move': move.uci() if move else None}
            else:
                uci = predict_next_move(
                    model, board,
                    temperature=float(req.get('temperature', 0.0)),
                    value_weight=float(req.get('value_weight', 2.0)),
                    value_candidate_limit=int(req.get('value_candidates', 0)),
                )
                response = {'move': uci}

            sys.stdout.write(json.dumps(response) + '\n')
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({
                'error': str(exc),
                'traceback': traceback.format_exc(),
            }) + '\n')
            sys.stdout.flush()

    return 0


if __name__ == '__main__':
    sys.exit(main())
