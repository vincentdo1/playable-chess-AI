"""MCTS/PUCT search using the trained network as policy prior and value evaluator.

AlphaZero-style inference search:
  - Policy head supplies the prior P(s, a) for each child move.
  - Value head supplies V(s) at newly-expanded leaves.
  - Tree descent uses PUCT:
        argmax_a [ -Q_child(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)) ]
    Q is negated because the child stores its own side-to-move's value, and the
    parent picks the move that's worst for the opponent.
  - Leaves are collected into batches and evaluated in one forward pass.
    Virtual loss penalises in-flight paths so parallel descents diverge.

The trained network's policy was learned one-hot from GM moves, so its
distribution over legal moves is very peaky. PUCT priors need to be a real
distribution for the U term to do useful exploration — so we apply a softening
temperature (default 1.5) to the logits before softmax. The value head is used
as trained.

Public entry point: ``mcts_search_best_move(model, board, ...)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import chess.polyglot
import numpy as np
import torch

from neural_network import (
    BOARD_ENCODING_VERSION, ChessModel, get_encoding_spec
)
from load_model import _model_encoding, _move_batch_tensor, _position_arrays


def _model_spec(model):
    return getattr(
        model, 'encoding_spec', get_encoding_spec(BOARD_ENCODING_VERSION)
    )


@dataclass
class MCTSStats:
    simulations: int = 0
    nn_batches: int = 0
    nn_evals: int = 0
    max_depth: int = 0
    terminal_hits: int = 0
    elapsed: float = 0.0
    stop_reason: str = 'not_started'


class MCTSNode:
    """Single node in the search tree. Slots keep per-node memory tight."""

    __slots__ = (
        'parent', 'move', 'children', 'prior',
        'visit_count', 'value_sum', 'virtual_loss', 'is_expanded',
    )

    def __init__(self, parent: 'MCTSNode | None' = None,
                 move: chess.Move | None = None, prior: float = 0.0):
        self.parent = parent
        self.move = move
        self.children: dict[chess.Move, MCTSNode] = {}
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.is_expanded = False

    def total_n(self) -> int:
        return self.visit_count + self.virtual_loss

    def q_value(self) -> float:
        """Mean value from this node's side-to-move POV, biased by virtual loss.

        Virtual loss is ADDED here (not subtracted) on purpose: selection reads
        this through negation (`-child.q_value()`), so inflating the child's own
        value makes the move look *worse to the parent*. That is what steers
        concurrent descents in a batch onto different paths. Subtracting would
        invert the penalty and collapse the batch onto one path.
        """
        n = self.total_n()
        if n == 0:
            return 0.0
        return (self.value_sum + self.virtual_loss) / n


def _expand_node(node: MCTSNode, board: chess.Board,
                 policy_logits: np.ndarray, policy_temperature: float,
                 move_to_index=None) -> None:
    """Create children for each legal move, with softened policy as priors."""
    if node.is_expanded:
        return
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        node.is_expanded = True
        return

    if move_to_index is None:
        move_to_index = get_encoding_spec(BOARD_ENCODING_VERSION)['move_to_index']
    is_black = (board.turn == chess.BLACK)
    legal_indices = np.array(
        [move_to_index(m, flip=is_black) for m in legal_moves],
        dtype=np.int64,
    )
    legal_logits = policy_logits[legal_indices] / max(policy_temperature, 1e-6)
    legal_logits = legal_logits - legal_logits.max()
    exp_logits = np.exp(legal_logits)
    total = exp_logits.sum()
    if total <= 0.0 or not np.isfinite(total):
        priors = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
    else:
        priors = exp_logits / total

    for move, prior in zip(legal_moves, priors):
        node.children[move] = MCTSNode(parent=node, move=move, prior=float(prior))
    node.is_expanded = True


def _select_child(node: MCTSNode, c_puct: float):
    """PUCT: pick child best for the parent's side to move."""
    parent_n = node.total_n()
    sqrt_parent_n = math.sqrt(max(parent_n, 1))

    best_score = -math.inf
    best_move = None
    best_child = None
    for move, child in node.children.items():
        q_for_parent = -child.q_value()
        u = c_puct * child.prior * sqrt_parent_n / (1 + child.total_n())
        score = q_for_parent + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child


def _terminal_value_stm(board: chess.Board) -> float | None:
    """Value from side-to-move POV if board is terminal, else None."""
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.can_claim_fifty_moves()
    ):
        return 0.0
    return None


def _backup(path: list[MCTSNode], value: float) -> None:
    """Walk path leaf→root, flipping value at each step. Releases virtual loss."""
    for node in reversed(path):
        if node.virtual_loss > 0:
            node.virtual_loss -= 1
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _release_virtual_loss(path: list[MCTSNode]) -> None:
    """Undo an in-flight descent that will not be evaluated."""
    for node in path:
        if node.virtual_loss > 0:
            node.virtual_loss -= 1


def _batched_eval(model: ChessModel, boards: list[chess.Board]):
    """Run one batched forward pass. Returns (policy_logits[B, V], values[B])."""
    if not boards:
        empty_p = np.zeros(
            (0, _model_spec(model)['move_vocab_size']), dtype=np.float32
        )
        empty_v = np.zeros(0, dtype=np.float32)
        return empty_p, empty_v

    device = next(model.parameters()).device
    encoding_version = _model_encoding(model)
    board_arrays = []
    move_arrays = []
    for board in boards:
        b, m = _position_arrays(board, encoding_version)
        board_arrays.append(b)
        move_arrays.append(m)

    board_batch = (
        torch.from_numpy(np.stack(board_arrays))
        .float()
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device, non_blocking=device.type == 'cuda')
    )
    move_batch = _move_batch_tensor(move_arrays, len(boards), device)

    with torch.inference_mode(), torch.amp.autocast(
        device.type, enabled=device.type == 'cuda'
    ):
        policy_logits, values = model(board_batch, move_batch)
    return (
        policy_logits.float().detach().cpu().numpy(),
        values.float().detach().cpu().numpy(),
    )


def _descend_to_leaf(root: MCTSNode, root_board: chess.Board, c_puct: float):
    """Walk down with PUCT, applying virtual loss. Returns (leaf, leaf_board, path, depth)."""
    node = root
    # Preserve the full reversible history. v2 consumes only the final ten
    # moves, but v3 repetition features and terminal draw detection cannot be
    # reconstructed soundly from an arbitrary fixed-length tail.
    board = root_board.copy(stack=True)
    path = [node]
    node.virtual_loss += 1
    depth = 0

    while node.is_expanded and node.children:
        move, child = _select_child(node, c_puct)
        if child is None:
            break
        board.push(move)
        child.virtual_loss += 1
        path.append(child)
        node = child
        depth += 1
    return node, board, path, depth


def mcts_search(
    model: ChessModel,
    root_board: chess.Board,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    batch_size: int = 8,
    policy_temperature: float = 1.5,
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    time_limit: float | None = None,
) -> tuple[MCTSNode, MCTSStats]:
    """Run PUCT simulations from root_board. Stops at num_simulations or time_limit."""
    if num_simulations < 0:
        raise ValueError('num_simulations must be >= 0')
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if time_limit is not None and (
        not math.isfinite(time_limit) or time_limit < 0
    ):
        raise ValueError('time_limit must be finite and >= 0')

    stats = MCTSStats()
    root = MCTSNode()
    start = time.monotonic()
    deadline = start + time_limit if time_limit and time_limit > 0 else None

    move_to_index = _model_spec(model)['move_to_index']

    # Pre-expand root.
    policies, values = _batched_eval(model, [root_board])
    stats.nn_batches += 1
    stats.nn_evals += 1
    _expand_node(root, root_board, policies[0], policy_temperature,
                 move_to_index=move_to_index)
    root.visit_count = 1
    root.value_sum = float(values[0])

    if add_root_noise and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * float(n)

    sims_done = 0
    timed_out = False
    while sims_done < num_simulations:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break

        batch_start_sims = sims_done
        target = min(batch_size, num_simulations - sims_done)
        pending: list[tuple[MCTSNode, chess.Board, list[MCTSNode]]] = []

        attempts = 0
        max_attempts = max(target * 4, 4)
        while (
            len(pending) + (sims_done - batch_start_sims) < target
            and attempts < max_attempts
        ):
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            attempts += 1
            leaf, leaf_board, path, depth = _descend_to_leaf(root, root_board, c_puct)
            stats.max_depth = max(stats.max_depth, depth)

            terminal_val = _terminal_value_stm(leaf_board)
            if terminal_val is not None:
                stats.terminal_hits += 1
                _backup(path, terminal_val)
                sims_done += 1
                if sims_done >= num_simulations:
                    break
                continue

            pending.append((leaf, leaf_board, path))

        if timed_out:
            for _, _, path in pending:
                _release_virtual_loss(path)
            break

        if not pending:
            if attempts >= max_attempts and sims_done < num_simulations:
                # All descents resolved as terminals this round; loop continues.
                continue
            break

        # A forward pass cannot be interrupted safely, but do not launch a new
        # batch after the deadline has already elapsed.
        if deadline is not None and time.monotonic() >= deadline:
            for _, _, path in pending:
                _release_virtual_loss(path)
            timed_out = True
            break

        boards_to_eval = [b for _, b, _ in pending]
        policies, values = _batched_eval(model, boards_to_eval)
        stats.nn_batches += 1
        stats.nn_evals += len(pending)

        for (leaf, leaf_board, path), policy, value in zip(pending, policies, values):
            _expand_node(leaf, leaf_board, policy, policy_temperature,
                         move_to_index=move_to_index)
            _backup(path, float(value))

        sims_done += len(pending)
        stats.simulations = sims_done

    stats.simulations = sims_done
    stats.elapsed = time.monotonic() - start
    if sims_done >= num_simulations:
        stats.stop_reason = 'simulation_limit'
    elif timed_out:
        stats.stop_reason = 'time_limit'
    else:
        stats.stop_reason = 'search_exhausted'
    return root, stats


def select_move_from_root(root: MCTSNode, temperature: float = 0.0) -> chess.Move | None:
    """Pick a root move from the visit distribution."""
    if not root.children:
        return None
    if not math.isfinite(temperature):
        raise ValueError('temperature must be finite')
    moves = list(root.children.keys())
    visits = np.array(
        [root.children[m].visit_count for m in moves], dtype=np.float64,
    )

    weights = visits
    if visits.sum() == 0:
        weights = np.array(
            [root.children[m].prior for m in moves], dtype=np.float64,
        )
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() == 0.0:
        weights = np.ones(len(moves), dtype=np.float64)
    if temperature <= 0.0:
        return moves[int(np.argmax(weights))]

    # Log-space scaling avoids all-zero underflow for tiny temperatures and
    # overflow when heavily visited children are raised to a large power.
    log_weights = np.full(len(weights), -np.inf, dtype=np.float64)
    positive = weights > 0.0
    log_weights[positive] = (
        np.log(weights[positive]) / max(temperature, 1e-6)
    )
    log_weights -= np.max(log_weights)
    pi = np.exp(log_weights)
    pi /= pi.sum()
    return moves[int(np.random.choice(len(moves), p=pi))]


def mcts_search_best_move_with_stats(
    model: ChessModel,
    board: chess.Board,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    batch_size: int = 8,
    policy_temperature: float = 1.5,
    move_temperature: float = 0.0,
    time_limit: float | None = None,
    verbose: bool = False,
) -> tuple[chess.Move | None, MCTSStats]:
    """Run MCTS and return both the chosen move and actual search stats."""
    if board.is_game_over(claim_draw=True):
        return None, MCTSStats(stop_reason='game_over')
    legal = list(board.legal_moves)
    if not legal:
        return None, MCTSStats(stop_reason='no_legal_moves')
    if len(legal) == 1:
        return legal[0], MCTSStats(stop_reason='single_legal_move')

    root, stats = mcts_search(
        model,
        board,
        num_simulations=num_simulations,
        c_puct=c_puct,
        batch_size=batch_size,
        policy_temperature=policy_temperature,
        time_limit=time_limit,
    )
    move = select_move_from_root(root, temperature=move_temperature)

    if verbose and move is not None:
        chosen = root.children[move]
        print(
            f"  MCTS picked {move.uci()} | N={chosen.visit_count} "
            f"Q={-chosen.q_value():+.3f} P={chosen.prior:.3f} | "
            f"sims={stats.simulations} batches={stats.nn_batches} "
            f"max_depth={stats.max_depth} terminals={stats.terminal_hits} "
            f"elapsed={stats.elapsed:.2f}s"
        )
        ranked = sorted(
            root.children.items(),
            key=lambda kv: kv[1].visit_count,
            reverse=True,
        )[:5]
        for m, c in ranked:
            print(
                f"    {m.uci():>6} N={c.visit_count:>5} "
                f"Q={-c.q_value():+.3f} P={c.prior:.3f}"
            )

    return move, stats


def mcts_search_best_move(
    model: ChessModel,
    board: chess.Board,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    batch_size: int = 8,
    policy_temperature: float = 1.5,
    move_temperature: float = 0.0,
    time_limit: float | None = None,
    verbose: bool = False,
) -> chess.Move | None:
    """Backward-compatible move-only MCTS entry point."""
    move, _ = mcts_search_best_move_with_stats(
        model,
        board,
        num_simulations=num_simulations,
        c_puct=c_puct,
        batch_size=batch_size,
        policy_temperature=policy_temperature,
        move_temperature=move_temperature,
        time_limit=time_limit,
        verbose=verbose,
    )
    return move


def mcts_predict_uci(
    model: ChessModel,
    board: chess.Board,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    batch_size: int = 8,
    policy_temperature: float = 1.5,
    move_temperature: float = 0.0,
    time_limit: float | None = None,
    verbose: bool = False,
) -> str | None:
    move = mcts_search_best_move(
        model, board,
        num_simulations=num_simulations,
        c_puct=c_puct,
        batch_size=batch_size,
        policy_temperature=policy_temperature,
        move_temperature=move_temperature,
        time_limit=time_limit,
        verbose=verbose,
    )
    return move.uci() if move is not None else None


if __name__ == '__main__':
    from load_model import load_trained_model

    print("Loading model...")
    model = load_trained_model()

    print("\n--- MCTS 400 sims from startpos ---")
    board = chess.Board()
    mcts_search_best_move(model, board, num_simulations=400, verbose=True)

    print("\n--- MCTS 400 sims from a tactical middlegame ---")
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    mcts_search_best_move(model, board, num_simulations=400, verbose=True)
