"""Flask backend — serves alphabeta and Magnus Carlsen moves. Stockfish runs client-side."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess_player
import os

app = Flask(__name__)
CORS(app)

_magnus_model = None
_predict_fn   = None

try:
    from load_model import load_trained_model, predict_next_move_with_search as predict_next_move
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
    """POST {fen, player, depth?} → {move, player}. player is 'alphabeta' or 'magnus'."""
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
        uci = _predict_fn(_magnus_model, board, top_n=10, depth=4)
        if uci is None:
            return jsonify({'error': 'Magnus model returned no move'}), 500
        return jsonify({'move': uci, 'player': 'magnus'})

    return jsonify({'error': f"Unknown player: '{player}'"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)