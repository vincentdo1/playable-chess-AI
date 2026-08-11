"""Flask backend for Alphabeta and Magnus moves. Stockfish runs client-side."""

import os

import chess
import chess_player
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

_magnus_model = None
_magnus_model_path = None
_predict_fn = None

DEFAULT_MAGNUS_TEMPERATURE = float(os.environ.get('MAGNUS_TEMPERATURE', '0.0'))
DEFAULT_MAGNUS_VALUE_WEIGHT = float(os.environ.get('MAGNUS_VALUE_WEIGHT', '2.0'))
DEFAULT_MAGNUS_VALUE_CANDIDATES = int(os.environ.get('MAGNUS_VALUE_CANDIDATES', '0'))


# MCTS for /api/move magnus. Off by default; enable with MAGNUS_USE_MCTS=1.
def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {'1', 'true', 'yes', 'on'}


DEFAULT_MAGNUS_USE_MCTS = _env_flag('MAGNUS_USE_MCTS', default=False)
# Shallow-search veto of material-losing policy moves (policy mode only).
DEFAULT_MAGNUS_BLUNDER_GUARD = _env_flag('MAGNUS_BLUNDER_GUARD', default=True)
DEFAULT_MAGNUS_BLUNDER_GUARD_DEPTH = int(os.environ.get('MAGNUS_BLUNDER_GUARD_DEPTH', '2'))
DEFAULT_MAGNUS_BLUNDER_GUARD_MARGIN = float(os.environ.get('MAGNUS_BLUNDER_GUARD_MARGIN', '150'))
DEFAULT_MAGNUS_MCTS_SIMULATIONS = int(os.environ.get('MAGNUS_MCTS_SIMULATIONS', '200'))
# Hard ceiling on request-supplied sims so a public endpoint can't be forced
# into unbounded CPU work. Raise via env if you run on stronger hardware.
DEFAULT_MAGNUS_MCTS_MAX_SIMULATIONS = int(os.environ.get('MAGNUS_MCTS_MAX_SIMULATIONS', '800'))
DEFAULT_MAGNUS_MCTS_BATCH = int(os.environ.get('MAGNUS_MCTS_BATCH', '16'))
DEFAULT_MAGNUS_MCTS_C_PUCT = float(os.environ.get('MAGNUS_MCTS_C_PUCT', '1.5'))
DEFAULT_MAGNUS_MCTS_POLICY_TEMP = float(os.environ.get('MAGNUS_MCTS_POLICY_TEMP', '1.5'))
# Wall-clock cap per move (seconds); MCTS stops at whichever of sims/time comes
# first. 0 = no cap. Set on CPU deploys (Railway) so a move can't run long.
DEFAULT_MAGNUS_MCTS_TIME_LIMIT = float(os.environ.get('MAGNUS_MCTS_TIME_LIMIT', '0'))

try:
    from load_model import MODEL_PATH, load_trained_model, predict_next_move

    _magnus_model_path = MODEL_PATH
    _magnus_model = load_trained_model()
    _predict_fn = predict_next_move
    print(f"Magnus Carlsen model loaded: {_magnus_model_path}")
except Exception as e:
    print(f"Magnus model unavailable: {e}")

# Optional MCTS; falls back to raw policy if the module isn't importable.
_mcts_fn = None
try:
    from inference.mcts_player import mcts_search_best_move as _mcts_fn
    if DEFAULT_MAGNUS_USE_MCTS:
        print(f"Magnus MCTS enabled: {DEFAULT_MAGNUS_MCTS_SIMULATIONS} sims/move")
except Exception as e:
    if DEFAULT_MAGNUS_USE_MCTS:
        print(f"MAGNUS_USE_MCTS requested but mcts_player unavailable ({e}); "
              "falling back to raw policy.")


@app.route('/')
def health():
    return jsonify({
        'status': 'ok',
        'players': {
            'alphabeta': True,
            'magnus': _magnus_model is not None,
        },
        'magnus': {
            'model': os.path.basename(_magnus_model_path or ''),
            'temperature': DEFAULT_MAGNUS_TEMPERATURE,
            'value_weight': DEFAULT_MAGNUS_VALUE_WEIGHT,
            'value_candidates': DEFAULT_MAGNUS_VALUE_CANDIDATES,
            'blunder_guard': DEFAULT_MAGNUS_BLUNDER_GUARD,
            'use_mcts': DEFAULT_MAGNUS_USE_MCTS and _mcts_fn is not None,
            'mcts_simulations': DEFAULT_MAGNUS_MCTS_SIMULATIONS,
            'mcts_time_limit': DEFAULT_MAGNUS_MCTS_TIME_LIMIT or None,
            'mcts_available': _mcts_fn is not None,
        },
    })


@app.route('/api/move', methods=['POST'])
def get_move():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    fen = data.get('fen')
    player = data.get('player')
    if not fen or not player:
        return jsonify({'error': "'fen' and 'player' are required"}), 400

    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f'Invalid FEN: {e}'}), 400

    if board.is_game_over():
        return jsonify({'error': 'Game is already over'}), 400

    if player == 'alphabeta':
        response, status = _get_alphabeta_move(board, data)
        return jsonify(response), status

    if player == 'magnus':
        response, status = _get_magnus_move(board, data)
        return jsonify(response), status

    return jsonify({'error': f"Unknown player: '{player}'"}), 400


def _get_alphabeta_move(board, data):
    try:
        depth = int(data.get('depth', 3))
    except (TypeError, ValueError):
        return {'error': "'depth' must be an integer"}, 400
    depth = max(1, min(depth, 4))

    move = chess_player.alphabeta(board.turn, board, depth)
    if move is None:
        return {'error': 'Alphabeta found no move'}, 500
    return {'move': move.uci(), 'player': 'alphabeta'}, 200


def _get_magnus_move(board, data):
    if _magnus_model is None:
        return {'error': 'Magnus model is not loaded on this server'}, 503

    try:
        temperature = float(data.get('temperature', DEFAULT_MAGNUS_TEMPERATURE))
    except (TypeError, ValueError):
        return {'error': "'temperature' must be a number"}, 400
    temperature = max(0.0, min(temperature, 3.0))

    try:
        value_weight = float(data.get('value_weight', DEFAULT_MAGNUS_VALUE_WEIGHT))
        value_candidates = int(data.get(
            'value_candidates', DEFAULT_MAGNUS_VALUE_CANDIDATES
        ))
    except (TypeError, ValueError):
        return {
            'error': "'value_weight' must be a number and "
                     "'value_candidates' must be an integer"
        }, 400
    value_candidates = max(0, value_candidates)

    # Per-request override of the MCTS toggle; defaults to the server's env flag.
    use_mcts = bool(data.get('use_mcts', DEFAULT_MAGNUS_USE_MCTS))
    if use_mcts and _mcts_fn is not None:
        try:
            simulations = int(data.get('mcts_simulations',
                                       DEFAULT_MAGNUS_MCTS_SIMULATIONS))
        except (TypeError, ValueError):
            return {'error': "'mcts_simulations' must be an integer"}, 400
        simulations = max(1, min(simulations, DEFAULT_MAGNUS_MCTS_MAX_SIMULATIONS))
        time_limit = (DEFAULT_MAGNUS_MCTS_TIME_LIMIT
                      if DEFAULT_MAGNUS_MCTS_TIME_LIMIT > 0 else None)

        move = _mcts_fn(
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
            'mcts_simulations': simulations,
            'mcts_time_limit': time_limit,
            'temperature': temperature,
        }, 200

    blunder_guard = bool(data.get('blunder_guard', DEFAULT_MAGNUS_BLUNDER_GUARD))
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
