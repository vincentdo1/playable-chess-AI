"""
prediction_model.py — DEPRECATED.

This file's functionality has been replaced by load_model.py, which uses
the trained PyTorch model to predict moves.

Use predict_next_move() from load_model.py instead:

    from load_model import load_trained_model, predict_next_move
    import chess

    model = load_trained_model()
    board = chess.Board()
    move  = predict_next_move(model, board)
    print(move)  # e.g. 'e2e4'
"""

raise ImportError(
    "prediction_model.py is deprecated. Use load_model.py instead.\n"
    "See the docstring in this file for usage."
)