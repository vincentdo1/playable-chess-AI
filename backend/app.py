"""Flask backend for Alphabeta and Magnus moves. Stockfish runs client-side."""

import math
import os
import threading

import chess
import chess_player
from flask import Flask, jsonify, request
from flask_cors import CORS


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(
        f'{name} must be a boolean (0/1, false/true, no/yes, off/on)'
    )


def _env_int(name, default, minimum=None, maximum=None):
    value = int(os.environ.get(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return value


def _env_float(name, default, minimum=None, maximum=None):
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return value


def _request_bool(data, name, default):
    if name not in data:
        return default
    value = data[name]
    if type(value) is not bool:
        raise ValueError(f"'{name}' must be a boolean")
    return value


def _request_float(data, name, default):
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"'{name}' must be finite")
    return value


def _request_int(data, name, default):
    value = data.get(name, default)
    if type(value) is not int:
        raise ValueError(f"'{name}' must be an integer")
    return value


DEFAULT_ALLOWED_ORIGINS = (
    'https://vincentdo1.github.io,'
    'http://localhost:8000,http://127.0.0.1:8000,null'
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'MAGNUS_ALLOWED_ORIGINS', DEFAULT_ALLOWED_ORIGINS
    ).split(',')
    if origin.strip()
]
MAX_REQUEST_BYTES = _env_int(
    'MAX_REQUEST_BYTES', 16 * 1024, minimum=1024, maximum=1024 * 1024
)
INFERENCE_MAX_CONCURRENCY = _env_int(
    'INFERENCE_MAX_CONCURRENCY', 1, minimum=1, maximum=32
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_REQUEST_BYTES
CORS(app, origins=ALLOWED_ORIGINS)

_inference_slots = threading.BoundedSemaphore(INFERENCE_MAX_CONCURRENCY)
_magnus_model = None
_magnus_model_path = None
_magnus_load_error = None
_predict_fn = None

DEFAULT_MAGNUS_TEMPERATURE = _env_float(
    'MAGNUS_TEMPERATURE', 0.0, minimum=0.0, maximum=3.0
)
DEFAULT_MAGNUS_VALUE_WEIGHT = _env_float(
    'MAGNUS_VALUE_WEIGHT', 2.0, minimum=-100.0, maximum=100.0
)
DEFAULT_MAGNUS_VALUE_CANDIDATES = _env_int(
    'MAGNUS_VALUE_CANDIDATES', 0, minimum=0, maximum=512
)


# MCTS for /api/move magnus. Off by default; enable with MAGNUS_USE_MCTS=1.
DEFAULT_MAGNUS_USE_MCTS = _env_flag('MAGNUS_USE_MCTS', default=False)
# Shallow-search veto of material-losing policy moves (policy mode only).
DEFAULT_MAGNUS_BLUNDER_GUARD = _env_flag('MAGNUS_BLUNDER_GUARD', default=True)
DEFAULT_MAGNUS_BLUNDER_GUARD_DEPTH = _env_int(
    'MAGNUS_BLUNDER_GUARD_DEPTH', 2, minimum=1, maximum=6
)
DEFAULT_MAGNUS_BLUNDER_GUARD_MARGIN = _env_float(
    'MAGNUS_BLUNDER_GUARD_MARGIN', 150, minimum=0, maximum=10000
)
DEFAULT_MAGNUS_MCTS_SIMULATIONS = _env_int(
    'MAGNUS_MCTS_SIMULATIONS', 200, minimum=1, maximum=100000
)
# Hard ceiling on request-supplied sims so a public endpoint can't be forced
# into unbounded CPU work. Raise via env if you run on stronger hardware.
DEFAULT_MAGNUS_MCTS_MAX_SIMULATIONS = _env_int(
    'MAGNUS_MCTS_MAX_SIMULATIONS', 800, minimum=1, maximum=100000
)
DEFAULT_MAGNUS_MCTS_BATCH = _env_int(
    'MAGNUS_MCTS_BATCH', 16, minimum=1, maximum=256
)
DEFAULT_MAGNUS_MCTS_C_PUCT = _env_float(
    'MAGNUS_MCTS_C_PUCT', 1.5, minimum=0.0, maximum=100.0
)
DEFAULT_MAGNUS_MCTS_POLICY_TEMP = _env_float(
    'MAGNUS_MCTS_POLICY_TEMP', 1.5, minimum=1e-6, maximum=100.0
)
# Wall-clock cap per move (seconds); MCTS stops at whichever of sims/time comes
# first. 0 = no cap. Set on CPU deploys (Railway) so a move can't run long.
DEFAULT_MAGNUS_MCTS_TIME_LIMIT = _env_float(
    'MAGNUS_MCTS_TIME_LIMIT', 0, minimum=0.0, maximum=120.0
)
MAGNUS_MCTS_SIMULATION_LIMIT = min(
    DEFAULT_MAGNUS_MCTS_SIMULATIONS,
    DEFAULT_MAGNUS_MCTS_MAX_SIMULATIONS,
)
_explicit_model_config = bool(
    os.environ.get('MODEL_PATH') or os.environ.get('MAGNUS_HF_REPO')
)
MAGNUS_REQUIRED = _env_flag(
    'MAGNUS_REQUIRED', default=_explicit_model_config
)

try:
    from load_model import MODEL_PATH, load_trained_model, predict_next_move

    _magnus_model_path = MODEL_PATH
    _magnus_model = load_trained_model()
    _predict_fn = predict_next_move
    print(f"Neural-network model loaded: {_magnus_model_path}")
except Exception as e:
    _magnus_load_error = e
    print(f"Magnus model unavailable: {e}", flush=True)
    if MAGNUS_REQUIRED:
        raise RuntimeError(
            'Magnus is required but its model could not be loaded'
        ) from e

# Optional MCTS; falls back to raw policy if the module isn't importable.
_mcts_fn = None
try:
    from inference.mcts_player import (
        mcts_search_best_move_with_stats as _mcts_fn,
    )
    if DEFAULT_MAGNUS_USE_MCTS:
        print(
            f"Magnus MCTS enabled: {MAGNUS_MCTS_SIMULATION_LIMIT} sims/move"
        )
except Exception as e:
    if DEFAULT_MAGNUS_USE_MCTS:
        print(f"MAGNUS_USE_MCTS requested but mcts_player unavailable ({e}); "
              "falling back to raw policy.")

if MAGNUS_REQUIRED and DEFAULT_MAGNUS_USE_MCTS and _mcts_fn is None:
    raise RuntimeError('Magnus MCTS is enabled and required, but unavailable')


def _magnus_ready():
    model_ready = _magnus_model is not None and _predict_fn is not None
    mcts_ready = not DEFAULT_MAGNUS_USE_MCTS or _mcts_fn is not None
    return model_ready and mcts_ready


def _readiness_payload():
    ready = not MAGNUS_REQUIRED or _magnus_ready()
    return {
        'status': 'ok' if ready else 'unavailable',
        'ready': ready,
        'players': {
            'alphabeta': True,
            'magnus': _magnus_model is not None,
        },
        'magnus': {
            'required': MAGNUS_REQUIRED,
            'ready': _magnus_ready(),
            'load_error': _magnus_load_error is not None,
            'model': os.path.basename(_magnus_model_path or ''),
            'architecture': getattr(_magnus_model, 'arch_version', None),
            'encoding': getattr(_magnus_model, 'encoding_version', None),
            'checkpoint_sha256': getattr(
                _magnus_model, 'checkpoint_sha256', None
            ),
            'hf_repo': getattr(_magnus_model, 'hf_repo', None),
            'hf_revision': getattr(_magnus_model, 'hf_revision', None),
            'temperature': DEFAULT_MAGNUS_TEMPERATURE,
            'value_weight': DEFAULT_MAGNUS_VALUE_WEIGHT,
            'value_candidates': DEFAULT_MAGNUS_VALUE_CANDIDATES,
            'blunder_guard': DEFAULT_MAGNUS_BLUNDER_GUARD,
            'use_mcts': DEFAULT_MAGNUS_USE_MCTS and _mcts_fn is not None,
            'mcts_simulations': MAGNUS_MCTS_SIMULATION_LIMIT,
            'mcts_time_limit': DEFAULT_MAGNUS_MCTS_TIME_LIMIT or None,
            'mcts_available': _mcts_fn is not None,
        },
    }, 200 if ready else 503


@app.route('/')
def health():
    payload, status = _readiness_payload()
    return jsonify(payload), status


@app.route('/readyz')
def readiness():
    payload, status = _readiness_payload()
    return jsonify(payload), status


@app.route('/livez')
def liveness():
    return jsonify({'status': 'ok'}), 200


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': f'JSON body exceeds {MAX_REQUEST_BYTES} bytes'}), 413


@app.route('/api/move', methods=['POST'])
def get_move():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({'error': 'JSON body required'}), 400

    fen = data.get('fen')
    player = data.get('player')
    if (
        not isinstance(fen, str)
        or not isinstance(player, str)
        or not fen
        or not player
    ):
        return jsonify({'error': "'fen' and 'player' are required"}), 400

    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f'Invalid FEN: {e}'}), 400

    if board.is_game_over():
        return jsonify({'error': 'Game is already over'}), 400

    if player not in {'alphabeta', 'magnus'}:
        return jsonify({'error': f"Unknown player: '{player}'"}), 400

    if not _inference_slots.acquire(blocking=False):
        response = jsonify({
            'error': 'Server is busy computing another move; retry shortly'
        })
        response.headers['Retry-After'] = '1'
        return response, 429
    try:
        if player == 'alphabeta':
            response, status = _get_alphabeta_move(board, data)
        else:
            response, status = _get_magnus_move(board, data)
        return jsonify(response), status
    finally:
        _inference_slots.release()


def _get_alphabeta_move(board, data):
    try:
        depth = _request_int(data, 'depth', 3)
    except ValueError as exc:
        return {'error': str(exc)}, 400
    depth = max(1, min(depth, 4))

    move = chess_player.alphabeta(board.turn, board, depth)
    if move is None:
        return {'error': 'Alphabeta found no move'}, 500
    return {'move': move.uci(), 'player': 'alphabeta'}, 200


def _get_magnus_move(board, data):
    if _magnus_model is None or _predict_fn is None:
        return {'error': 'Magnus model is not loaded on this server'}, 503

    try:
        temperature = _request_float(
            data, 'temperature', DEFAULT_MAGNUS_TEMPERATURE
        )
    except ValueError as exc:
        return {'error': str(exc)}, 400
    temperature = max(0.0, min(temperature, 3.0))

    try:
        value_weight = _request_float(
            data, 'value_weight', DEFAULT_MAGNUS_VALUE_WEIGHT
        )
        if not -100.0 <= value_weight <= 100.0:
            raise ValueError("'value_weight' must be between -100 and 100")
        value_candidates = _request_int(
            data,
            'value_candidates', DEFAULT_MAGNUS_VALUE_CANDIDATES
        )
        if value_candidates < 0 or value_candidates > 512:
            raise ValueError("'value_candidates' must be between 0 and 512")
        requested_mcts = _request_bool(
            data, 'use_mcts', DEFAULT_MAGNUS_USE_MCTS
        )
    except ValueError as exc:
        return {'error': str(exc)}, 400

    # The server flag is an allow-list, not merely a default. A client may opt
    # into a cheaper policy move, but cannot turn expensive MCTS on server-side.
    use_mcts = DEFAULT_MAGNUS_USE_MCTS and requested_mcts
    if use_mcts and _mcts_fn is not None:
        try:
            requested_simulations = _request_int(
                data, 'mcts_simulations', MAGNUS_MCTS_SIMULATION_LIMIT
            )
            if requested_simulations < 1:
                raise ValueError("'mcts_simulations' must be >= 1")
        except ValueError as exc:
            return {'error': str(exc)}, 400
        simulations = min(
            requested_simulations, MAGNUS_MCTS_SIMULATION_LIMIT
        )
        time_limit = (DEFAULT_MAGNUS_MCTS_TIME_LIMIT
                      if DEFAULT_MAGNUS_MCTS_TIME_LIMIT > 0 else None)

        move, stats = _mcts_fn(
            _magnus_model,
            board,
            num_simulations=simulations,
            c_puct=DEFAULT_MAGNUS_MCTS_C_PUCT,
            batch_size=DEFAULT_MAGNUS_MCTS_BATCH,
            policy_temperature=DEFAULT_MAGNUS_MCTS_POLICY_TEMP,
            move_temperature=temperature,
            time_limit=time_limit,
        )
        if move is None:
            return {'error': 'Magnus MCTS returned no move'}, 500
        return {
            'move': move.uci(),
            'player': 'magnus',
            'method': 'mcts',
            'mcts_simulations': stats.simulations,
            'mcts_simulations_requested': requested_simulations,
            'mcts_simulations_budget': simulations,
            'mcts_time_limit': time_limit,
            'mcts_elapsed': round(stats.elapsed, 4),
            'mcts_stop_reason': stats.stop_reason,
            'temperature': temperature,
        }, 200

    try:
        blunder_guard = _request_bool(
            data, 'blunder_guard', DEFAULT_MAGNUS_BLUNDER_GUARD
        )
    except ValueError as exc:
        return {'error': str(exc)}, 400
    uci = _predict_fn(
        _magnus_model,
        board,
        temperature=temperature,
        value_weight=value_weight,
        value_candidate_limit=value_candidates,
        blunder_guard=blunder_guard,
        blunder_guard_depth=DEFAULT_MAGNUS_BLUNDER_GUARD_DEPTH,
        blunder_guard_margin_cp=DEFAULT_MAGNUS_BLUNDER_GUARD_MARGIN,
    )
    if uci is None:
        return {'error': 'Magnus model returned no move'}, 500

    return {
        'move': uci,
        'player': 'magnus',
        'method': 'policy',
        'temperature': temperature,
        'value_weight': value_weight,
        'value_candidates': value_candidates,
        'blunder_guard': blunder_guard,
    }, 200


def run():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    run()
