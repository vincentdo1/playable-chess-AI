"""Load a trained ChessModel and predict the next move."""

import hashlib
import os
import re

import numpy as np
import torch
import chess

from neural_network import (
    BOARD_ENCODING_VERSION, BOARD_ENCODING_VERSION_V3, ChessModel,
    ChessModelV3, ChessModelV4, board_to_tensor_v3, fen_to_tensor,
    get_encoding_spec, move_sequence_to_vector, MODEL_PATH, DEVICE
)

DEFAULT_VALUE_WEIGHT = float(os.environ.get('MAGNUS_VALUE_WEIGHT', '2.0'))
DEFAULT_VALUE_CANDIDATES = int(os.environ.get('MAGNUS_VALUE_CANDIDATES', '0'))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'on'}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hf_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    normalized = revision.strip().lower()
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', normalized):
        raise ValueError(
            'MAGNUS_HF_REVISION must be a 40- or 64-character hexadecimal '
            'commit ID; branch/tag names such as "main" are mutable.'
        )
    return normalized


def _maybe_fetch_from_hf(path: str) -> str:
    """Return a local path to the checkpoint, downloading it from a Hugging
    Face model repo if `path` is missing and MAGNUS_HF_REPO is set.

    This is how the Railway/container deploy gets the git-ignored .pt: the
    weights live in an HF model repo and are pulled on first boot into the HF
    cache. Local dev is unaffected -- if the file already exists we never touch
    the network or import huggingface_hub.

    Env:
      MAGNUS_HF_REPO      repo id, e.g. "vincentdo/magnus-chess-v4" (required)
      MAGNUS_HF_FILENAME  file in the repo (default: basename of `path`)
      MAGNUS_HF_REVISION  immutable commit SHA (required by default)
      HF_TOKEN            only for a private repo

    Set MAGNUS_ALLOW_FLOATING_HF_REVISION=1 only for local experimentation.
    """
    if os.path.exists(path):
        return path
    repo_id = os.environ.get('MAGNUS_HF_REPO')
    if not repo_id:
        return path  # let the caller raise the normal FileNotFoundError
    filename = os.environ.get('MAGNUS_HF_FILENAME', os.path.basename(path))
    revision = os.environ.get('MAGNUS_HF_REVISION') or None
    allow_floating = _env_flag(
        'MAGNUS_ALLOW_FLOATING_HF_REVISION', default=False
    )
    if not allow_floating:
        if revision is None:
            raise ValueError(
                'MAGNUS_HF_REVISION must pin an immutable Hugging Face commit '
                'when MAGNUS_HF_REPO is used. Set '
                'MAGNUS_ALLOW_FLOATING_HF_REVISION=1 only for an intentional '
                'development download from the latest revision.'
            )
        revision = _validate_hf_revision(revision)
    token = os.environ.get('HF_TOKEN') or None
    from huggingface_hub import hf_hub_download  # lazy: only needed to download
    print(f"Model {path!r} not found locally; fetching {filename!r} from "
          f"Hugging Face repo {repo_id!r}...")
    local_path = hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, token=token
    )
    print(f"Downloaded checkpoint to {local_path}")
    return local_path


def _checkpoint_dimension(checkpoint, key: str, default: int,
                          maximum: int) -> int:
    value = int(checkpoint.get(key, default))
    if not 1 <= value <= maximum:
        raise ValueError(f'Invalid checkpoint {key}={value!r}')
    return value


def load_trained_model(path: str = MODEL_PATH) -> ChessModel:
    """Load a saved ChessModel from a .pt checkpoint file."""
    used_hf_download = (
        not os.path.exists(path) and bool(os.environ.get('MAGNUS_HF_REPO'))
    )
    path = _maybe_fetch_from_hf(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model checkpoint {path!r} was not found. Train the current "
            "perspective/residual model or set MODEL_PATH to a matching "
            "checkpoint, or set MAGNUS_HF_REPO to fetch it from Hugging Face."
        )

    checkpoint_sha256 = _sha256_file(path)
    expected_sha256 = os.environ.get('MAGNUS_MODEL_SHA256', '').lower().strip()
    if expected_sha256:
        if len(expected_sha256) != 64 or any(
            char not in '0123456789abcdef' for char in expected_sha256
        ):
            raise ValueError('MAGNUS_MODEL_SHA256 must be 64 hexadecimal characters')
        if checkpoint_sha256 != expected_sha256:
            raise RuntimeError(
                f'Checkpoint SHA-256 mismatch for {path!r}: expected '
                f'{expected_sha256}, got {checkpoint_sha256}'
            )

    # Explicit safe loading avoids the arbitrary-code-capable pickle behavior
    # used by older PyTorch defaults. Project checkpoints contain only tensors
    # and primitive metadata, all supported by weights_only=True.
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(f'Checkpoint {path!r} has no model_state_dict')
    # All generations stay servable. arch_version picks the architecture
    # (v4 SE-ResNet shares the v3 board encoding, so encoding alone is not
    # enough); older checkpoints without arch_version fall back to the
    # stored board encoding (v2 ResNet+LSTM or v3 conv-head ResNet).
    checkpoint_encoding = checkpoint.get('board_encoding') or BOARD_ENCODING_VERSION
    arch_version = checkpoint.get('arch_version')
    if arch_version not in {None, 'v2', 'v3', 'v4'}:
        raise ValueError(f'Checkpoint uses unknown architecture {arch_version!r}.')
    expected_encoding = {
        'v2': BOARD_ENCODING_VERSION,
        'v3': BOARD_ENCODING_VERSION_V3,
        'v4': BOARD_ENCODING_VERSION_V3,
    }.get(arch_version)
    if expected_encoding is not None and checkpoint_encoding != expected_encoding:
        raise ValueError(
            f'Checkpoint architecture {arch_version!r} requires board encoding '
            f'{expected_encoding!r}, got {checkpoint_encoding!r}.'
        )

    filters = _checkpoint_dimension(
        checkpoint, 'residual_filters', 256 if arch_version == 'v4' else 128,
        maximum=1024,
    )
    blocks = _checkpoint_dimension(
        checkpoint, 'residual_blocks', 12 if arch_version == 'v4' else 8,
        maximum=128,
    )
    if arch_version == 'v4':
        model = ChessModelV4(
            filters=filters,
            blocks=blocks,
        ).to(DEVICE)
    elif checkpoint_encoding == BOARD_ENCODING_VERSION_V3:
        model = ChessModelV3(filters=filters, blocks=blocks).to(DEVICE)
    elif checkpoint_encoding == BOARD_ENCODING_VERSION:
        model = ChessModel(filters=filters, blocks=blocks).to(DEVICE)
    else:
        raise ValueError(
            f"Checkpoint uses unknown board encoding {checkpoint_encoding!r}."
        )
    try:
        load_result = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load {path!r} into the current model architecture. "
            "This usually means the checkpoint was trained before the "
            "perspective/residual architecture change."
        ) from exc
    # A missing tensor means that layer would be served with random init.
    # The one legitimate case is an old policy-only checkpoint with no value
    # head at all — that loads with value reranking disabled. Anything else
    # (a partial/corrupt file, an architecture drift) must fail loudly
    # instead of silently playing with a partially random network.
    missing_value_head = [
        key for key in load_result.missing_keys
        if key.startswith('value_head')
    ]
    other_missing = [
        key for key in load_result.missing_keys
        if not key.startswith('value_head')
    ]
    checkpoint_has_value_head = any(
        key.startswith('value_head') for key in checkpoint['model_state_dict']
    )
    if (
        other_missing or
        (missing_value_head and checkpoint_has_value_head) or
        load_result.unexpected_keys
    ):
        raise RuntimeError(
            f"Checkpoint {path!r} does not cover the current architecture; "
            "refusing to serve a mismatched network. "
            f"Missing tensors: {load_result.missing_keys}; unexpected tensors: "
            f"{load_result.unexpected_keys}"
        )
    model.eval()
    model.arch_version = arch_version or (
        'v3' if checkpoint_encoding == BOARD_ENCODING_VERSION_V3 else 'v2'
    )
    model.encoding_version = checkpoint_encoding
    model.encoding_spec = get_encoding_spec(checkpoint_encoding)
    model.checkpoint_sha256 = checkpoint_sha256
    model.checkpoint_path = path
    model.hf_repo = (
        os.environ.get('MAGNUS_HF_REPO') if used_hf_download else None
    )
    model.hf_revision = (
        os.environ.get('MAGNUS_HF_REVISION') if used_hf_download else None
    )

    # Loading weights is not enough to prove the serving contract is usable.
    # Run one cheap forward pass with the checkpoint-selected encoding before
    # readiness can turn green; channel/vocabulary drift then fails at startup
    # instead of on the first user request.
    spec = model.encoding_spec
    board_probe = torch.zeros(
        1, spec['board_channels'], 8, 8, device=DEVICE
    )
    history_probe = (
        torch.zeros(1, 10, 132, device=DEVICE)
        if spec['uses_move_history']
        else torch.zeros(1, 0, device=DEVICE)
    )
    try:
        with torch.no_grad():
            policy_probe, value_probe = model(board_probe, history_probe)
    except Exception as exc:
        raise RuntimeError(
            f'Checkpoint {path!r} loaded but failed its startup smoke forward.'
        ) from exc
    if (
        tuple(policy_probe.shape) != (1, spec['move_vocab_size']) or
        tuple(value_probe.shape) != (1,) or
        not bool(torch.isfinite(policy_probe).all()) or
        not bool(torch.isfinite(value_probe).all())
    ):
        raise RuntimeError(
            f'Checkpoint {path!r} produced an invalid smoke-forward result: '
            f'policy={tuple(policy_probe.shape)}, value={tuple(value_probe.shape)}.'
        )
    print(f"Model loaded from {path}  (encoding: {checkpoint_encoding})")
    print(f"  Architecture: {model.arch_version}  |  SHA-256: {checkpoint_sha256}")
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss')
    if val_loss is None:
        print(f"  Saved at epoch {epoch}")
    else:
        print(f"  Saved at epoch {epoch}  |  val_loss={val_loss:.4f}")
    model.value_head_trained = not missing_value_head
    if not model.value_head_trained:
        print("  Value reranking disabled because this checkpoint has no trained value head.")
    return model

def _model_encoding(model) -> str:
    return getattr(model, 'encoding_version', BOARD_ENCODING_VERSION)

def _position_arrays(board: chess.Board,
                     encoding_version: str = BOARD_ENCODING_VERSION):
    """Board (+ history for v2) arrays for one position.

    v3 returns None for the move-history array: that architecture is
    board-only, which also removes the v2 train/serve skew where FEN-built
    boards had no move stack to feed the LSTM.
    """
    is_black = (board.turn == chess.BLACK)
    if encoding_version == BOARD_ENCODING_VERSION_V3:
        return board_to_tensor_v3(board, flip=is_black), None
    return (
        fen_to_tensor(board.fen(), flip=is_black),
        move_sequence_to_vector(
            list(board.move_stack[-10:]), max_length=10, flip=is_black
        ),
    )

def _move_batch_tensor(move_arrays, batch_size, device):
    """Stack v2 history arrays, or build the empty v3 placeholder."""
    if move_arrays and move_arrays[0] is not None:
        return torch.tensor(
            np.stack(move_arrays), dtype=torch.float32
        ).to(device)
    return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

def _position_tensors(model: ChessModel, board: chess.Board):
    board_tensor, move_seq = _position_arrays(board, _model_encoding(model))
    model_device = next(model.parameters()).device
    board_t = (
        torch.tensor(board_tensor, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(model_device)
    )
    move_t = _move_batch_tensor(
        [move_seq] if move_seq is not None else [], 1, model_device
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

    encoding_version = _model_encoding(model)
    scores = np.zeros(len(moves), dtype=np.float32)
    pending_indices = []
    board_tensors = []
    move_tensors = []

    for idx, move in enumerate(moves):
        board.push(move)
        terminal_value = _terminal_value_for_previous_mover(board)
        if terminal_value is None:
            board_tensor, move_seq = _position_arrays(board, encoding_version)
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
        move_batch = _move_batch_tensor(
            move_tensors, len(pending_indices), model_device
        )
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

    move_to_index = getattr(
        model, 'encoding_spec', get_encoding_spec(BOARD_ENCODING_VERSION)
    )['move_to_index']
    move_indices = torch.tensor(
        [move_to_index(move, flip=is_black) for move in legal_moves],
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
                      value_candidate_limit: int | None = DEFAULT_VALUE_CANDIDATES,
                      blunder_guard: bool = False,
                      blunder_guard_depth: int = 2,
                      blunder_guard_margin_cp: float = 150.0,
                      ) -> str | None:
    """
    Pure neural network prediction with policy+value scoring.

    temperature controls variety:
      0.0 = always pick the single best move (greedy)
      1.2 = natural variety while still preferring confident moves
      2.0 = very exploratory

    value_weight controls how strongly resulting-position value reranks moves.
    value_candidate_limit=0 or None evaluates every legal move with the value head.
    blunder_guard shallow-searches the top candidates and vetoes ones that
    lose material (inference/blunder_guard.py).
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
    if blunder_guard and scored:
        from inference.blunder_guard import filter_scored_moves
        scored = filter_scored_moves(
            board, scored,
            depth=blunder_guard_depth,
            margin_cp=blunder_guard_margin_cp,
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
