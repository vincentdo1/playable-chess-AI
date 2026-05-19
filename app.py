"""Flask backend - serves alphabeta and Magnus Carlsen moves. Stockfish runs client-side."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess_player
import os

app = Flask(__name__)
CORS(app)

_magnus_model = None
_predict_fn   = None
DEFAULT_MAGNUS_TEMPERATURE = float(os.environ.get('MAGNUS_TEMPERATURE', '1.2'))
DEFAULT_MAGNUS_VALUE_WEIGHT = float(os.environ.get('MAGNUS_VALUE_WEIGHT', '2.0'))
DEFAULT_MAGNUS_VALUE_CANDIDATES = int(os.environ.get('MAGNUS_VALUE_CANDIDATES', '0'))

try:
    from load_model import load_trained_model, predict_next_move
    _magnus_model = load_trained_model()
    _predict_fn   = predict_next_move
    print("Magnus Carlsen model loaded.")
except Exception as e:
    print(f"Magnus model unavailable: {e}")

# Routes

@app.route('/')
def health():
    """Health check — also reports which players are available."""
    return jsonify({
        'status': 'ok',
        'players': {
            'alphabeta': True,
            'magnus':    _magnus_model is not None,
        }
    })

@app.route('/api/move', methods=['POST'])
def get_move():
    """POST {fen, player, depth?, temperature?} -> {move, player}."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    fen    = data.get('fen')
    player = data.get('player')
    depth  = min(int(data.get('depth', 3)), 4)  # cap at 4 to avoid timeouts

    if not fen or not player:
        return jsonify({'error': "'fen' and 'player' are required"}), 400

    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({'error': f'Invalid FEN: {e}'}), 400

    if board.is_game_over():
        return jsonify({'error': 'Game is already over'}), 400

    if player == 'alphabeta':
        move = chess_player.alphabeta(board.turn, board, depth)
        if move is None:
            return jsonify({'error': 'Alphabeta found no move'}), 500
        return jsonify({'move': move.uci(), 'player': 'alphabeta'})

    if player == 'magnus':
        if _magnus_model is None:
            return jsonify({'error': 'Magnus model is not loaded on this server'}), 503
        try:
            temperature = float(data.get('temperature', DEFAULT_MAGNUS_TEMPERATURE))
        except (TypeError, ValueError):
            return jsonify({'error': "'temperature' must be a number"}), 400
        temperature = max(0.0, min(temperature, 3.0))
        try:
            value_weight = float(data.get('value_weight', DEFAULT_MAGNUS_VALUE_WEIGHT))
            value_candidates = int(data.get(
                'value_candidates', DEFAULT_MAGNUS_VALUE_CANDIDATES
            ))
        except (TypeError, ValueError):
            return jsonify({
                'error': "'value_weight' must be a number and "
                         "'value_candidates' must be an integer"
            }), 400
        value_candidates = max(0, value_candidates)

        uci = _predict_fn(
            _magnus_model,
            board,
            temperature=temperature,
            value_weight=value_weight,
            value_candidate_limit=value_candidates,
        )
        if uci is None:
            return jsonify({'error': 'Magnus model returned no move'}), 500
        return jsonify({
            'move': uci,
            'player': 'magnus',
            'temperature': temperature,
            'value_weight': value_weight,
            'value_candidates': value_candidates,
        })

    return jsonify({'error': f"Unknown player: '{player}'"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
