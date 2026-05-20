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


def _choose_move(player, board, magnus_model, magnus_temperature,
                 magnus_value_weight, magnus_value_candidates,
                 alphabeta_depth):
    if player == 'magnus':
        uci = predict_next_move(
            magnus_model,
            board,
            temperature=magnus_temperature,
            value_weight=magnus_value_weight,
            value_candidate_limit=magnus_value_candidates,
        )
        return chess.Move.from_uci(uci) if uci else None
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
              magnus_value_candidates, alphabeta_depth, max_plies):
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
    parser.add_argument('--max_plies', type=int, default=160)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--output_dir', default='analysis_games')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=1)
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
            if args.magnus_color == 'white':
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
