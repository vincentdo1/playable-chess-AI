"""
app.py — Flask backend for the Chess AI web interface.

Exposes two Python-based AI players via HTTP:
  - Alphabeta pruning  (chess_player.py)
  - Magnus Carlsen NN  (load_model.py + PyTorch)

Stockfish and Random run client-side in JS — no API call needed.

Deploy to Railway (free):
  1. Push this repo to GitHub
  2. Go to railway.app → New Project → Deploy from GitHub repo
  3. Railway auto-detects Python and runs this file
  4. Paste the generated URL into index.html as API_URL

Local development:
  pip install flask flask-cors
  python app.py
  Server starts at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess_player
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load Magnus model once at startup — graceful if not yet trained
# ---------------------------------------------------------------------------
_magnus_model = None
_predict_fn   = None

try:
    from load_model import load_trained_model, predict_next_move
    _magnus_model = load_trained_model()
    _predict_fn   = predict_next_move
    print("Magnus Carlsen model loaded.")
except Exception as e:
    print(f"Magnus model unavailable: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def health():
    """
    Health check. Also tells the frontend which players are available
    so it can enable/disable the Magnus option in the UI dynamically.
    """
    return jsonify({
        'status': 'ok',
        'players': {
            'alphabeta': True,
            'magnus':    _magnus_model is not None,
        }
    })


@app.route('/api/move', methods=['POST'])
def get_move():
    """
    Get the next move for a Python-based AI player.

    Request JSON:
        fen    (str)  — board position in FEN notation
        player (str)  — 'alphabeta' or 'magnus'
        depth  (int)  — search depth for alphabeta (1-4, default 3)

    Response JSON:
        { move: 'e2e4', player: 'alphabeta' }
    """
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

    # --- Alphabeta ---
    if player == 'alphabeta':
        move = chess_player.alphabeta(board.turn, board, depth)
        if move is None:
            return jsonify({'error': 'Alphabeta found no move'}), 500
        return jsonify({'move': move.uci(), 'player': 'alphabeta'})

    # --- Magnus Carlsen neural network ---
    if player == 'magnus':
        if _magnus_model is None:
            return jsonify({'error': 'Magnus model is not loaded on this server'}), 503
        uci = _predict_fn(_magnus_model, board)
        if uci is None:
            return jsonify({'error': 'Magnus model returned no move'}), 500
        return jsonify({'move': uci, 'player': 'magnus'})

    return jsonify({'error': f"Unknown player: '{player}'"}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)