"""Run headless engine matches and save PGNs for model review."""

import argparse
import os
import random
from datetime import datetime

import chess
import chess.pgn
import numpy as np

import chess_player
from load_model import load_trained_model, predict_next_move
from neural_network import MODEL_PATH
from inference.search_player import search_best_move
from inference.mcts_player import mcts_search_best_move


def _choose_move(player, board, magnus_model, magnus_temperature,
                 magnus_value_weight, magnus_value_candidates,
                 alphabeta_depth, search_depth, search_time,
                 mcts_simulations, mcts_c_puct, mcts_batch_size,
                 mcts_policy_temperature, mcts_move_temperature,
                 mcts_time):
    if player == 'magnus':
        uci = predict_next_move(
            magnus_model,
            board,
            temperature=magnus_temperature,
            value_weight=magnus_value_weight,
            value_candidate_limit=magnus_value_candidates,
        )
        return chess.Move.from_uci(uci) if uci else None
    if player == 'magnus_search':
        return search_best_move(
            magnus_model,
            board,
            max_depth=search_depth,
            time_limit=search_time,
        )
    if player == 'magnus_mcts':
        return mcts_search_best_move(
            magnus_model,
            board,
            num_simulations=mcts_simulations,
            c_puct=mcts_c_puct,
            batch_size=mcts_batch_size,
            policy_temperature=mcts_policy_temperature,
            move_temperature=mcts_move_temperature,
            time_limit=mcts_time,
        )
    if player == 'alphabeta':
        return chess_player.alphabeta(board.turn, board, alphabeta_depth)
    if player == 'random':
        return chess_player.random_move_player(board)
    raise ValueError(f"Unknown player: {player}")


def _result_for_board(board, terminated_by_move_limit):
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.result(), outcome.termination.name.lower()
    if terminated_by_move_limit:
        return '*', 'move_limit'
    return '*', 'unfinished'


def play_game(game_number, white_player, black_player, magnus_model,
              magnus_temperature, magnus_value_weight,
              magnus_value_candidates, alphabeta_depth,
              search_depth, search_time,
              mcts_simulations, mcts_c_puct, mcts_batch_size,
              mcts_policy_temperature, mcts_move_temperature, mcts_time,
              max_plies):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers['Event'] = 'Magnus NN vs Alphabeta'
    game.headers['Site'] = 'playable-chess-AI local match'
    game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')
    game.headers['Round'] = str(game_number)
    game.headers['White'] = white_player
    game.headers['Black'] = black_player

    node = game
    terminated_by_move_limit = False
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= max_plies:
            terminated_by_move_limit = True
            break

        player = white_player if board.turn == chess.WHITE else black_player
        move = _choose_move(
            player,
            board,
            magnus_model,
            magnus_temperature,
            magnus_value_weight,
            magnus_value_candidates,
            alphabeta_depth,
            search_depth,
            search_time,
            mcts_simulations,
            mcts_c_puct,
            mcts_batch_size,
            mcts_policy_temperature,
            mcts_move_temperature,
            mcts_time,
        )
        if move is None or move not in board.legal_moves:
            game.headers['Termination'] = f'{player}_illegal_or_no_move'
            game.headers['Result'] = '0-1' if board.turn == chess.WHITE else '1-0'
            return game, game.headers['Result']

        board.push(move)
        node = node.add_variation(move)

    result, termination = _result_for_board(board, terminated_by_move_limit)
    game.headers['Result'] = result
    game.headers['Termination'] = termination
    game.headers['PlyCount'] = str(len(board.move_stack))
    game.headers['MagnusTemperature'] = str(magnus_temperature)
    game.headers['MagnusValueWeight'] = str(magnus_value_weight)
    game.headers['MagnusValueCandidates'] = str(magnus_value_candidates)
    game.headers['AlphabetaDepth'] = str(alphabeta_depth)
    game.headers['SearchDepth'] = str(search_depth)
    if search_time is not None:
        game.headers['SearchTime'] = str(search_time)
    game.headers['MCTSSimulations'] = str(mcts_simulations)
    game.headers['MCTSCPuct'] = str(mcts_c_puct)
    game.headers['MCTSBatchSize'] = str(mcts_batch_size)
    game.headers['MCTSPolicyTemperature'] = str(mcts_policy_temperature)
    game.headers['MCTSMoveTemperature'] = str(mcts_move_temperature)
    if mcts_time is not None:
        game.headers['MCTSTime'] = str(mcts_time)
    return game, result


def default_output_path(output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_dir, f'magnus_vs_alphabeta_{timestamp}.pgn')


def main():
    parser = argparse.ArgumentParser(
        description='Play Magnus NN vs alphabeta games and save PGN.'
    )
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--alphabeta_depth', type=int, default=2)
    parser.add_argument('--magnus_temperature', type=float, default=1.2)
    parser.add_argument('--magnus_value_weight', type=float, default=2.0)
    parser.add_argument('--magnus_value_candidates', type=int, default=0)
    parser.add_argument('--search_depth', type=int, default=4,
                        help='Max plies for magnus_search alpha-beta.')
    parser.add_argument('--search_time', type=float, default=None,
                        help='Seconds per move for magnus_search '
                             '(stops iterative deepening early).')
    parser.add_argument('--mcts_simulations', type=int, default=400,
                        help='Total PUCT simulations per move for magnus_mcts.')
    parser.add_argument('--mcts_c_puct', type=float, default=1.5,
                        help='PUCT exploration constant.')
    parser.add_argument('--mcts_batch_size', type=int, default=8,
                        help='Leaves per batched NN forward pass.')
    parser.add_argument('--mcts_policy_temperature', type=float, default=1.5,
                        help='Softening applied to policy logits before using '
                             'them as PUCT priors (>1 softens a peaky policy).')
    parser.add_argument('--mcts_move_temperature', type=float, default=0.0,
                        help='Sampling temperature on root visits. 0 = greedy.')
    parser.add_argument('--mcts_time', type=float, default=None,
                        help='Seconds-per-move cap for magnus_mcts; stops early '
                             'even if --mcts_simulations is not yet reached.')
    parser.add_argument('--max_plies', type=int, default=160)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--output_dir', default='analysis_games')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--white',
        choices=('magnus', 'magnus_search', 'magnus_mcts', 'alphabeta', 'random'),
        default=None,
        help='Override white player (otherwise uses --magnus_color rotation).',
    )
    parser.add_argument(
        '--black',
        choices=('magnus', 'magnus_search', 'magnus_mcts', 'alphabeta', 'random'),
        default=None,
        help='Override black player (otherwise uses --magnus_color rotation).',
    )
    parser.add_argument(
        '--magnus_color',
        choices=('white', 'black', 'alternate'),
        default='alternate',
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.magnus_value_candidates = max(0, args.magnus_value_candidates)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output or default_output_path(args.output_dir)

    magnus_model = load_trained_model(args.model)
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}

    with open(output_path, 'w', encoding='utf-8') as pgn_file:
        for game_number in range(1, args.games + 1):
            if args.white is not None and args.black is not None:
                white_player, black_player = args.white, args.black
            elif args.magnus_color == 'white':
                white_player, black_player = 'magnus', 'alphabeta'
            elif args.magnus_color == 'black':
                white_player, black_player = 'alphabeta', 'magnus'
            elif game_number % 2 == 1:
                white_player, black_player = 'magnus', 'alphabeta'
            else:
                white_player, black_player = 'alphabeta', 'magnus'

            game, result = play_game(
                game_number,
                white_player,
                black_player,
                magnus_model,
                args.magnus_temperature,
                args.magnus_value_weight,
                args.magnus_value_candidates,
                args.alphabeta_depth,
                args.search_depth,
                args.search_time,
                args.mcts_simulations,
                args.mcts_c_puct,
                args.mcts_batch_size,
                args.mcts_policy_temperature,
                args.mcts_move_temperature,
                args.mcts_time,
                args.max_plies,
            )
            results[result] = results.get(result, 0) + 1
            print(game, file=pgn_file, end='\n\n')
            print(f"Game {game_number}: {white_player} vs {black_player} -> "
                  f"{result} ({game.headers.get('Termination')}, "
                  f"{game.headers.get('PlyCount', '?')} plies)")

    print()
    print(f"Saved PGN: {output_path}")
    print("Results:", ', '.join(f"{k}={v}" for k, v in results.items()))


if __name__ == '__main__':
    main()
