"""Measure the model against Stockfish at a fixed skill level / movetime.

Mirrors the browser matchup (stockfish.js `Skill Level`, `go movetime`) so a
"lost to level N in the UI" report can be reproduced and quantified headlessly:

  python -m evaluation.vs_stockfish --skill 10 --movetime 0.6 --games 6 \
      --mode policy            # what the backend plays by default
  python -m evaluation.vs_stockfish --skill 10 --movetime 0.6 --games 6 \
      --mode mcts --sims 200   # the UI's MCTS toggle

Modes reproduce the backend's serving configs, including the blunder guard
in policy mode. Colors alternate; Skill Level's built-in randomization keeps
games diverse even though the model plays greedily.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import chess
import chess.engine
import chess.pgn

from load_model import load_trained_model, predict_next_move
from inference.mcts_player import mcts_search_best_move
from neural_network import MODEL_PATH

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', 'stockfish.exe')
if not os.path.isabs(STOCKFISH_PATH) and os.path.exists(STOCKFISH_PATH):
    STOCKFISH_PATH = os.path.abspath(STOCKFISH_PATH)


def model_move(model, board, args):
    if args.mode == 'policy':
        uci = predict_next_move(
            model, board,
            temperature=args.temperature,
            value_weight=args.value_weight,
            value_candidate_limit=args.value_candidates,
            blunder_guard=not args.no_blunder_guard,
            blunder_guard_depth=args.guard_depth,
        )
        return chess.Move.from_uci(uci) if uci else None
    if args.mode == 'mcts':
        return mcts_search_best_move(
            model, board,
            num_simulations=args.sims,
            batch_size=args.mcts_batch_size,
            policy_temperature=args.mcts_policy_temp,
            move_temperature=0.0,
        )
    raise ValueError(f'unknown mode {args.mode!r}')


def play_game(model, engine, args, model_is_white: bool) -> tuple[str, int]:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= args.max_plies:
            return '1/2-1/2', len(board.move_stack)
        if (board.turn == chess.WHITE) == model_is_white:
            move = model_move(model, board, args)
        else:
            move = engine.play(
                board, chess.engine.Limit(time=args.movetime)
            ).move
        if move is None or move not in board.legal_moves:
            # A None/illegal move loses on the spot (same rule as eval_arena).
            return ('0-1' if board.turn == chess.WHITE else '1-0',
                    len(board.move_stack))
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    return outcome.result() if outcome else '1/2-1/2', len(board.move_stack)


def main():
    parser = argparse.ArgumentParser(
        description='Model vs Stockfish at a fixed Skill Level.'
    )
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--mode', choices=('policy', 'mcts'), default='policy')
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--skill', type=int, default=10,
                        help='Stockfish Skill Level 0-20.')
    parser.add_argument('--uci_elo', type=int, default=None,
                        help='Use calibrated UCI_LimitStrength/UCI_Elo '
                             '(1320-3190) instead of Skill Level. This is '
                             'the mode to use for absolute Elo estimates.')
    parser.add_argument('--movetime', type=float, default=0.6,
                        help='Seconds per Stockfish move (browser uses 0.6).')
    parser.add_argument('--max_plies', type=int, default=300)
    # policy-mode knobs (backend defaults)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--value_weight', type=float, default=2.0)
    parser.add_argument('--value_candidates', type=int, default=0)
    parser.add_argument('--no_blunder_guard', action='store_true')
    parser.add_argument('--guard_depth', type=int, default=2)
    # mcts-mode knobs (backend defaults)
    parser.add_argument('--sims', type=int, default=200)
    parser.add_argument('--mcts_batch_size', type=int, default=16)
    parser.add_argument('--mcts_policy_temp', type=float, default=1.5)
    args = parser.parse_args()

    model = load_trained_model(args.model)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    if args.uci_elo is not None:
        elo = max(1320, min(args.uci_elo, 3190))
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': elo})
        opponent = f'SF UCI_Elo {elo}'
    else:
        engine.configure({'Skill Level': args.skill})
        opponent = f'SF skill {args.skill}'

    label = args.mode if args.mode == 'policy' else f'mcts-{args.sims}'
    print(f'{label} vs {opponent} '
          f'@ {args.movetime * 1000:.0f}ms/move, {args.games} games',
          flush=True)

    score = 0.0
    wins = draws = losses = 0
    try:
        for g in range(args.games):
            model_is_white = (g % 2 == 0)
            start = datetime.now()
            result, plies = play_game(model, engine, args, model_is_white)
            if result == '1/2-1/2':
                draws += 1
                score += 0.5
            elif (result == '1-0') == model_is_white:
                wins += 1
                score += 1.0
            else:
                losses += 1
            elapsed = (datetime.now() - start).total_seconds()
            print(f'  game {g + 1}/{args.games}: model as '
                  f'{"White" if model_is_white else "Black"} -> {result} '
                  f'({plies} plies, {elapsed:.0f}s) '
                  f'[running {wins}W-{draws}D-{losses}L]', flush=True)
    finally:
        engine.quit()

    n = wins + draws + losses
    print(f'\n{label} vs {opponent}: {wins}W-{draws}D-{losses}L  '
          f'score {score}/{n} ({score / max(n, 1):.1%})', flush=True)


if __name__ == '__main__':
    main()
