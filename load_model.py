"""Load a trained ChessModel and predict the next move."""

import os

import numpy as np
import torch
import chess

from neural_network import (ChessModel, fen_to_tensor, move_sequence_to_vector,
                             move_to_policy_index, MODEL_PATH, DEVICE)

LEGACY_MODEL_PATH = 'model/grandmaster_model_policy_v1.pt'
DEFAULT_VALUE_WEIGHT = float(os.environ.get('MAGNUS_VALUE_WEIGHT', '2.0'))
DEFAULT_VALUE_CANDIDATES = int(os.environ.get('MAGNUS_VALUE_CANDIDATES', '0'))

def load_trained_model(path: str = MODEL_PATH) -> ChessModel:
    """Load a saved ChessModel from a .pt checkpoint file."""
    if (
        path == MODEL_PATH and
        not os.path.exists(path) and
        os.path.exists(LEGACY_MODEL_PATH)
    ):
        print(f"Model {path} not found; falling back to {LEGACY_MODEL_PATH}")
        path = LEGACY_MODEL_PATH

    checkpoint = torch.load(path, map_location=DEVICE)
    model = ChessModel().to(DEVICE)
    load_result = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    model.eval()
    print(f"Model loaded from {path}")
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss')
    if val_loss is None:
        print(f"  Saved at epoch {epoch}")
    else:
        print(f"  Saved at epoch {epoch}  |  val_loss={val_loss:.4f}")
    if load_result.missing_keys:
        print(f"  Warning: initialized missing layers: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  Warning: ignored checkpoint layers: {load_result.unexpected_keys}")
    model.value_head_trained = not any(
        key.startswith('value_head') for key in load_result.missing_keys
    )
    if not model.value_head_trained:
        print("  Value reranking disabled because this checkpoint has no trained value head.")
    return model

def _position_arrays(board: chess.Board):
    is_black = (board.turn == chess.BLACK)
    return (
        fen_to_tensor(board.fen(), flip=is_black),
        move_sequence_to_vector(
            list(board.move_stack[-10:]), max_length=10, flip=is_black
        ),
    )

def _position_tensors(model: ChessModel, board: chess.Board):
    board_tensor, move_seq = _position_arrays(board)
    model_device = next(model.parameters()).device
    board_t = (
        torch.tensor(board_tensor, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(model_device)
    )
    move_t = (
        torch.tensor(move_seq, dtype=torch.float32)
        .unsqueeze(0)
        .to(model_device)
    )
    return board_t, move_t

def _terminal_value_for_previous_mover(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    previous_mover = not board.turn
    return 1.0 if outcome.winner == previous_mover else -1.0

def _candidate_limit_to_int(value_candidate_limit):
    if value_candidate_limit is None:
        return None
    limit = int(value_candidate_limit)
    return limit if limit > 0 else None

def _value_scores_after_moves(model: ChessModel, board: chess.Board, moves):
    """Return resulting-position values from the current player's perspective."""
    if not moves:
        return np.array([], dtype=np.float32)

    scores = np.zeros(len(moves), dtype=np.float32)
    pending_indices = []
    board_tensors = []
    move_tensors = []

    for idx, move in enumerate(moves):
        board.push(move)
        terminal_value = _terminal_value_for_previous_mover(board)
        if terminal_value is None:
            board_tensor, move_seq = _position_arrays(board)
            board_tensors.append(board_tensor)
            move_tensors.append(move_seq)
            pending_indices.append(idx)
        else:
            scores[idx] = terminal_value
        board.pop()

    if pending_indices:
        model_device = next(model.parameters()).device
        board_batch = (
            torch.tensor(np.stack(board_tensors), dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .to(model_device)
        )
        move_batch = torch.tensor(
            np.stack(move_tensors), dtype=torch.float32
        ).to(model_device)
        with torch.no_grad():
            _, value_pred = model(board_batch, move_batch)
        current_player_values = -value_pred.detach().cpu().numpy()
        for idx, value in zip(pending_indices, current_player_values):
            scores[idx] = float(value)

    return scores

def _get_move_scores(model: ChessModel, board: chess.Board,
                     value_weight: float = DEFAULT_VALUE_WEIGHT,
                     value_candidate_limit: int | None = DEFAULT_VALUE_CANDIDATES):
    """
    Run the neural network and return a score for every legal move.

    Returns a list of (score, move) tuples sorted highest score first.
    Scores combine legal move policy with resulting-position value.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    is_black = (board.turn == chess.BLACK)
    board_t, move_t = _position_tensors(model, board)
    with torch.no_grad():
        policy_logits, _ = model(board_t, move_t)

    move_indices = torch.tensor(
        [move_to_policy_index(move, flip=is_black) for move in legal_moves],
        dtype=torch.long,
        device=policy_logits.device,
    )
    policy_scores = torch.log_softmax(
        policy_logits[0, move_indices], dim=0
    ).detach().cpu().numpy()

    combined_scores = policy_scores.copy()
    if (
        value_weight != 0.0 and
        getattr(model, 'value_head_trained', True)
    ):
        candidate_limit = _candidate_limit_to_int(value_candidate_limit)
        if candidate_limit is None or candidate_limit >= len(legal_moves):
            candidate_indices = np.arange(len(legal_moves))
        else:
            candidate_indices = np.argsort(policy_scores)[-candidate_limit:]

        candidate_moves = [legal_moves[i] for i in candidate_indices]
        candidate_values = _value_scores_after_moves(model, board, candidate_moves)
        for idx, value in zip(candidate_indices, candidate_values):
            combined_scores[idx] += value_weight * value

    scored = list(zip(combined_scores, legal_moves))
    scored.sort(key=lambda x: float(x[0]), reverse=True)
    return scored

def evaluate_position(model: ChessModel, board: chess.Board) -> float:
    """Return the model value estimate from the side-to-move's perspective."""
    board_t, move_t = _position_tensors(model, board)
    with torch.no_grad():
        _, value_pred = model(board_t, move_t)

    return float(value_pred[0].item())

def predict_next_move(model: ChessModel, board: chess.Board,
                      temperature: float = 1.2,
                      value_weight: float = DEFAULT_VALUE_WEIGHT,
                      value_candidate_limit: int | None = DEFAULT_VALUE_CANDIDATES
                      ) -> str | None:
    """
    Pure neural network prediction with policy+value scoring.

    temperature controls variety:
      0.0 = always pick the single best move (greedy)
      1.2 = natural variety while still preferring confident moves
      2.0 = very exploratory

    value_weight controls how strongly resulting-position value reranks moves.
    value_candidate_limit=0 or None evaluates every legal move with the value head.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    scored = _get_move_scores(
        model,
        board,
        value_weight=value_weight,
        value_candidate_limit=value_candidate_limit,
    )
    log_scores = np.array([s for s, _ in scored])
    moves  = [m for _, m in scored]

    if temperature == 0.0:
        return moves[0].uci()

    log_scores = log_scores / temperature
    log_scores -= log_scores.max()   # subtract max for stability
    probs = np.exp(log_scores)
    probs /= probs.sum()

    chosen = np.random.choice(len(moves), p=probs)
    return moves[chosen].uci()

if __name__ == '__main__':
    print("Loading model...")
    model = load_trained_model()

    board = chess.Board()

    print("\n--- Pure NN prediction (deterministic) ---")
    move = predict_next_move(model, board, temperature=0.0)
    print(f"Predicted first move: {move}")

    print("\n--- Pure NN prediction (sampled) ---")
    move = predict_next_move(model, board, temperature=1.2)
    print(f"Predicted first move: {move}")

    print("\n--- Policy-only comparison ---")
    move = predict_next_move(model, board, temperature=0.0, value_weight=0.0)
    print(f"Policy-only first move: {move}")
