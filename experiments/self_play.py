"""Generate MCTS self-play positions, visit targets, and game outcomes.

Root noise and an early-game sampling temperature diversify play. Policy
targets contain legal moves only, and policy/value targets use the side-to-move
perspective expected by training and inference.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

import chess
import numpy as np
import torch

from neural_network import (
    BOARD_ENCODING_VERSION,
    ChessModel,
    MOVE_VOCAB_SIZE,
    get_encoding_spec,
)
from load_model import load_trained_model, _model_encoding, _position_arrays
from inference.mcts_player import mcts_search, select_move_from_root


@dataclass
class SelfPlayConfig:
    """All hyperparameters for one batch of self-play game generation."""
    # MCTS
    mcts_simulations: int = 400
    c_puct: float = 1.5
    mcts_batch_size: int = 16
    policy_temperature: float = 1.0  # softening of prior; 1.0 = use raw policy
    # Move selection schedule
    temperature_threshold: int = 30
    temperature_high: float = 1.0
    temperature_low: float = 0.0
    # Root exploration noise (AlphaZero defaults for chess)
    add_root_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Game length safety
    max_plies: int = 256


@dataclass
class GameRecord:
    """Per-position training data plus aggregate game metadata."""
    boards: np.ndarray            # (T, 8, 8, C) float32 — C per board encoding
    move_seqs: np.ndarray | None  # (T, 10, 132) float32 for v2; None for v3
    legal_move_indices: np.ndarray  # (M,) int32 — concatenated legal indices
    legal_move_offsets: np.ndarray  # (T+1,) int64 — offsets into legal_move_indices
    visit_counts: np.ndarray        # (M,) int32 — parallel to legal_move_indices
    value_targets: np.ndarray       # (T,) float32 in {-1, 0, +1}
    side_white: np.ndarray          # (T,) bool — True iff white to move at this step
    result: str                     # "1-0" / "0-1" / "1/2-1/2" / "*"
    termination: str
    plies: int


def play_self_play_game(
    model: ChessModel,
    config: SelfPlayConfig,
    rng: np.random.Generator | None = None,
) -> GameRecord:
    """Play a complete self-play game; return per-position training data."""
    if rng is None:
        rng = np.random.default_rng()

    board = chess.Board()
    encoding_version = _model_encoding(model)
    spec = get_encoding_spec(encoding_version)
    move_to_index = spec['move_to_index']

    boards_list: list[np.ndarray] = []
    move_seqs_list: list[np.ndarray] = []
    legal_indices_list: list[np.ndarray] = []
    visit_counts_list: list[np.ndarray] = []
    sides_white: list[bool] = []

    terminated_by_move_limit = False

    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) >= config.max_plies:
            terminated_by_move_limit = True
            break

        is_black = (board.turn == chess.BLACK)
        position_board, position_moves = _position_arrays(
            board, encoding_version
        )

        root, _ = mcts_search(
            model,
            board,
            num_simulations=config.mcts_simulations,
            c_puct=config.c_puct,
            batch_size=config.mcts_batch_size,
            policy_temperature=config.policy_temperature,
            add_root_noise=config.add_root_noise,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )

        if not root.children:
            break

        legal_indices_for_pos: list[int] = []
        visit_counts_for_pos: list[int] = []
        for move, child in root.children.items():
            legal_indices_for_pos.append(
                move_to_index(move, flip=is_black)
            )
            visit_counts_for_pos.append(child.visit_count)

        ply = len(board.move_stack)
        move_temp = (
            config.temperature_high
            if ply < config.temperature_threshold
            else config.temperature_low
        )
        move = select_move_from_root(root, temperature=move_temp)
        if move is None:
            break

        boards_list.append(position_board)
        if position_moves is not None:
            move_seqs_list.append(position_moves)
        legal_indices_list.append(
            np.asarray(legal_indices_for_pos, dtype=np.int32)
        )
        visit_counts_list.append(
            np.asarray(visit_counts_for_pos, dtype=np.int32)
        )
        sides_white.append(board.turn == chess.WHITE)

        board.push(move)

    if terminated_by_move_limit:
        result_str = '1/2-1/2'
        winner: chess.Color | None = None
        termination = 'move_limit'
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            result_str = '*'
            winner = None
            termination = 'unknown'
        else:
            result_str = outcome.result()
            winner = outcome.winner
            termination = outcome.termination.name.lower()

    T = len(boards_list)
    value_targets = np.zeros(T, dtype=np.float32)
    if winner is not None:
        for i, is_white in enumerate(sides_white):
            value_targets[i] = 1.0 if (winner == chess.WHITE) == is_white else -1.0
    uses_history = spec['uses_move_history']
    if T == 0:
        return GameRecord(
            boards=np.empty((0, 8, 8, spec['board_channels']), dtype=np.float32),
            move_seqs=(np.empty((0, 10, 132), dtype=np.float32)
                       if uses_history else None),
            legal_move_indices=np.empty(0, dtype=np.int32),
            legal_move_offsets=np.zeros(1, dtype=np.int64),
            visit_counts=np.empty(0, dtype=np.int32),
            value_targets=np.empty(0, dtype=np.float32),
            side_white=np.empty(0, dtype=bool),
            result=result_str,
            termination=termination,
            plies=len(board.move_stack),
        )

    offsets = np.zeros(T + 1, dtype=np.int64)
    for i, arr in enumerate(legal_indices_list):
        offsets[i + 1] = offsets[i] + len(arr)
    legal_move_indices = np.concatenate(legal_indices_list).astype(np.int32)
    visit_counts = np.concatenate(visit_counts_list).astype(np.int32)

    return GameRecord(
        boards=np.stack(boards_list).astype(np.float32),
        move_seqs=(np.stack(move_seqs_list).astype(np.float32)
                   if uses_history else None),
        legal_move_indices=legal_move_indices,
        legal_move_offsets=offsets,
        visit_counts=visit_counts,
        value_targets=value_targets,
        side_white=np.asarray(sides_white, dtype=bool),
        result=result_str,
        termination=termination,
        plies=len(board.move_stack),
    )


def _save_chunk(
    output_path: str,
    records: list[GameRecord],
    config: SelfPlayConfig,
    encoding_version: str = BOARD_ENCODING_VERSION,
) -> int:
    """Concatenate records into one .npz chunk. Returns number of positions saved."""
    spec = get_encoding_spec(encoding_version)
    extra = {}
    if not records:
        if spec['uses_move_history']:
            extra['move_seqs'] = np.empty((0, 10, 132), dtype=np.float32)
        np.savez_compressed(
            output_path,
            boards=np.empty(
                (0, 8, 8, spec['board_channels']), dtype=np.float32
            ),
            board_encoding=np.array(encoding_version, dtype=np.str_),
            legal_move_indices=np.empty(0, dtype=np.int32),
            legal_move_offsets=np.zeros(1, dtype=np.int64),
            visit_counts=np.empty(0, dtype=np.int32),
            value_target=np.empty(0, dtype=np.float32),
            **extra,
        )
        return 0

    boards = np.concatenate([r.boards for r in records], axis=0)
    if spec['uses_move_history']:
        extra['move_seqs'] = np.concatenate(
            [r.move_seqs for r in records], axis=0
        )
    value_targets = np.concatenate([r.value_targets for r in records], axis=0)

    # Offsets need to be rebased across the concatenated games.
    all_legal_indices: list[np.ndarray] = []
    all_visit_counts: list[np.ndarray] = []
    offsets_parts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    running = 0
    for r in records:
        all_legal_indices.append(r.legal_move_indices)
        all_visit_counts.append(r.visit_counts)
        # Rebase each game's sparse-policy offsets without duplicating zero.
        shifted = r.legal_move_offsets[1:] + running
        offsets_parts.append(shifted)
        running += len(r.legal_move_indices)
    legal_move_indices = np.concatenate(all_legal_indices).astype(np.int32)
    visit_counts = np.concatenate(all_visit_counts).astype(np.int32)
    legal_move_offsets = np.concatenate(offsets_parts).astype(np.int64)

    np.savez_compressed(
        output_path,
        boards=boards,
        board_encoding=np.array(encoding_version, dtype=np.str_),
        legal_move_indices=legal_move_indices,
        legal_move_offsets=legal_move_offsets,
        visit_counts=visit_counts,
        value_target=value_targets,
        **extra,
    )
    return len(boards)


def generate_self_play_games(
    model: ChessModel,
    num_games: int,
    config: SelfPlayConfig,
    output_dir: str,
    chunk_prefix: str = 'selfplay',
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Generate num_games self-play games and write one .npz chunk."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chunk_path = os.path.join(output_dir, f'{chunk_prefix}_{timestamp}.npz')

    model.eval()

    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
    records: list[GameRecord] = []
    total_plies = 0
    start = time.monotonic()

    for game_idx in range(num_games):
        game_start = time.monotonic()
        record = play_self_play_game(model, config, rng=rng)
        elapsed = time.monotonic() - game_start

        records.append(record)
        results[record.result] = results.get(record.result, 0) + 1
        total_plies += record.plies

        if verbose:
            n_positions = len(record.boards)
            print(
                f"  Game {game_idx + 1}/{num_games}: {record.result} "
                f"({record.termination}, {record.plies} plies, "
                f"{n_positions} positions saved, {elapsed:.1f}s)"
            )

    n_positions_total = _save_chunk(
        chunk_path, records, config,
        encoding_version=_model_encoding(model),
    )
    total_elapsed = time.monotonic() - start

    summary = {
        'chunk_path': chunk_path,
        'num_games': num_games,
        'num_positions': n_positions_total,
        'total_plies': total_plies,
        'elapsed_seconds': total_elapsed,
        'results': results,
        'config': asdict(config),
    }
    if verbose:
        print()
        print(f"Wrote {chunk_path}")
        print(f"  positions={n_positions_total} plies={total_plies}")
        print(f"  results={results}")
        print(
            f"  elapsed={total_elapsed:.1f}s "
            f"({total_elapsed/max(num_games,1):.1f}s/game)"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Generate self-play training chunks via MCTS-guided games.'
    )
    parser.add_argument('--games', type=int, default=10,
                        help='Number of self-play games to generate.')
    parser.add_argument('--output_dir', default='data/selfplay_chunks')
    parser.add_argument('--chunk_prefix', default='selfplay')
    parser.add_argument('--model', default=None,
                        help='Model checkpoint to play with; defaults to MODEL_PATH.')
    parser.add_argument('--mcts_simulations', type=int, default=400)
    parser.add_argument('--c_puct', type=float, default=1.5)
    parser.add_argument('--mcts_batch_size', type=int, default=16)
    parser.add_argument('--policy_temperature', type=float, default=1.0)
    parser.add_argument('--temperature_threshold', type=int, default=30)
    parser.add_argument('--temperature_high', type=float, default=1.0)
    parser.add_argument('--temperature_low', type=float, default=0.0)
    parser.add_argument('--no_root_noise', action='store_true',
                        help='Disable Dirichlet root noise (NOT recommended for '
                             'real self-play; diversity collapses without it).')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.25)
    parser.add_argument('--max_plies', type=int, default=256)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if args.model is None:
        model = load_trained_model()
    else:
        model = load_trained_model(args.model)

    config = SelfPlayConfig(
        mcts_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        mcts_batch_size=args.mcts_batch_size,
        policy_temperature=args.policy_temperature,
        temperature_threshold=args.temperature_threshold,
        temperature_high=args.temperature_high,
        temperature_low=args.temperature_low,
        add_root_noise=not args.no_root_noise,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        max_plies=args.max_plies,
    )

    generate_self_play_games(
        model,
        num_games=args.games,
        config=config,
        output_dir=args.output_dir,
        chunk_prefix=args.chunk_prefix,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
