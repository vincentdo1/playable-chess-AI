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

from neural_network import ChessModel, MOVE_VOCAB_SIZE, move_to_policy_index
from load_model import _position_arrays


@dataclass
class MCTSStats:
    simulations: int = 0
    nn_batches: int = 0
    nn_evals: int = 0
    max_depth: int = 0
    terminal_hits: int = 0
    elapsed: float = 0.0


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
        """Mean value from this node's side-to-move POV, penalised by virtual loss."""
        n = self.total_n()
        if n == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / n


def _expand_node(node: MCTSNode, board: chess.Board,
                 policy_logits: np.ndarray, policy_temperature: float) -> None:
    """Create children for each legal move, with softened policy as priors."""
    if node.is_expanded:
        return
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        node.is_expanded = True
        return

    is_black = (board.turn == chess.BLACK)
    legal_indices = np.array(
        [move_to_policy_index(m, flip=is_black) for m in legal_moves],
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


def _batched_eval(model: ChessModel, boards: list[chess.Board]):
    """Run one batched forward pass. Returns (policy_logits[B, V], values[B])."""
    if not boards:
        empty_p = np.zeros((0, MOVE_VOCAB_SIZE), dtype=np.float32)
        empty_v = np.zeros(0, dtype=np.float32)
        return empty_p, empty_v

    device = next(model.parameters()).device
    board_arrays = []
    move_arrays = []
    for board in boards:
        b, m = _position_arrays(board)
        board_arrays.append(b)
        move_arrays.append(m)

    board_batch = (
        torch.from_numpy(np.stack(board_arrays))
        .float()
        .permute(0, 3, 1, 2)
        .to(device)
    )
    move_batch = (
        torch.from_numpy(np.stack(move_arrays))
        .float()
        .to(device)
    )

    with torch.no_grad():
        policy_logits, values = model(board_batch, move_batch)
    return policy_logits.detach().cpu().numpy(), values.detach().cpu().numpy()


def _descend_to_leaf(root: MCTSNode, root_board: chess.Board, c_puct: float):
    """Walk down with PUCT, applying virtual loss. Returns (leaf, leaf_board, path, depth)."""
    node = root
    # Keep last 10 moves for the LSTM input; we only ever need the recent tail.
    board = root_board.copy(stack=10)
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
    stats = MCTSStats()
    root = MCTSNode()
    start = time.monotonic()
    deadline = start + time_limit if time_limit else None

    # Pre-expand root.
    policies, values = _batched_eval(model, [root_board])
    stats.nn_batches += 1
    stats.nn_evals += 1
    _expand_node(root, root_board, policies[0], policy_temperature)
    root.visit_count = 1
    root.value_sum = float(values[0])

    if add_root_noise and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * float(n)

    sims_done = 0
    while sims_done < num_simulations:
        if deadline is not None and time.monotonic() >= deadline:
            break

        target = min(batch_size, num_simulations - sims_done)
        pending: list[tuple[MCTSNode, chess.Board, list[MCTSNode]]] = []

        attempts = 0
        max_attempts = max(target * 4, 4)
        while len(pending) < target and attempts < max_attempts:
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

        if not pending:
            if attempts >= max_attempts and sims_done < num_simulations:
                # All descents resolved as terminals this round; loop continues.
                continue
            break

        boards_to_eval = [b for _, b, _ in pending]
        policies, values = _batched_eval(model, boards_to_eval)
        stats.nn_batches += 1
        stats.nn_evals += len(pending)

        for (leaf, leaf_board, path), policy, value in zip(pending, policies, values):
            _expand_node(leaf, leaf_board, policy, policy_temperature)
            _backup(path, float(value))

        sims_done += len(pending)
        stats.simulations = sims_done

    stats.simulations = sims_done
    stats.elapsed = time.monotonic() - start
    return root, stats


def select_move_from_root(root: MCTSNode, temperature: float = 0.0) -> chess.Move | None:
    """Pick a root move from the visit distribution."""
    if not root.children:
        return None
    moves = list(root.children.keys())
    visits = np.array(
        [root.children[m].visit_count for m in moves], dtype=np.float64,
    )

    if temperature == 0.0 or visits.sum() == 0:
        return moves[int(np.argmax(visits))]

    pi = visits ** (1.0 / max(temperature, 1e-6))
    pi /= pi.sum()
    return moves[int(np.random.choice(len(moves), p=pi))]


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
    """End-to-end: run MCTS and return the chosen move."""
    if board.is_game_over(claim_draw=True):
        return None
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

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
