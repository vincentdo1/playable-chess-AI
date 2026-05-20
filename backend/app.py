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

try:
    from load_model import MODEL_PATH, load_trained_model, predict_next_move

    _magnus_model_path = MODEL_PATH
    _magnus_model = load_trained_model()
    _predict_fn = predict_next_move
    print(f"Magnus Carlsen model loaded: {_magnus_model_path}")
except Exception as e:
    print(f"Magnus model unavailable: {e}")


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

    uci = _predict_fn(
        _magnus_model,
        board,
        temperature=temperature,
        value_weight=value_weight,
        value_candidate_limit=value_candidates,
    )
    if uci is None:
        return {'error': 'Magnus model returned no move'}, 500

    return {
        'move': uci,
        'player': 'magnus',
        'temperature': temperature,
        'value_weight': value_weight,
        'value_candidates': value_candidates,
    }, 200


def run():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    run()
