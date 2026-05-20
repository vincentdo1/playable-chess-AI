import unittest

import chess
import torch

from neural_network import (
    BOARD_CHANNELS,
    MOVE_VOCAB_SIZE,
    ChessModel,
    collate_policy_batch,
    move_to_policy_index,
)
from load_model import (
    _get_move_scores,
    _terminal_value_for_previous_mover,
    predict_next_move,
)


class ValueHeadTest(unittest.TestCase):
    def test_model_returns_policy_logits_and_value(self):
        model = ChessModel()
        boards = torch.zeros((2, BOARD_CHANNELS, 8, 8), dtype=torch.float32)
        moves = torch.zeros((2, 10, 132), dtype=torch.float32)

        policy_logits, value_pred = model(boards, moves)

        self.assertEqual(policy_logits.shape, (2, MOVE_VOCAB_SIZE))
        self.assertEqual(value_pred.shape, (2,))
        self.assertTrue(torch.all(value_pred >= -1.0))
        self.assertTrue(torch.all(value_pred <= 1.0))

    def test_collate_policy_batch_includes_value_targets(self):
        move_idx = move_to_policy_index(chess.Move.from_uci('e2e4'))
        board = torch.zeros((BOARD_CHANNELS, 8, 8), dtype=torch.float32)
        moves = torch.zeros((10, 132), dtype=torch.float32)
        target = torch.tensor(move_idx, dtype=torch.long)
        legal = torch.tensor([move_idx], dtype=torch.long)
        weight = torch.tensor(0.7, dtype=torch.float32)
        value = torch.tensor(0.25, dtype=torch.float32)
        target_type = torch.tensor(1, dtype=torch.int8)

        batch = collate_policy_batch([
            (board, moves, target, legal, weight, value, target_type),
            (board, moves, target, legal, weight, value, target_type),
        ])

        self.assertEqual(len(batch), 7)
        self.assertEqual(batch[5].shape, (2,))
        self.assertTrue(torch.allclose(batch[5], torch.tensor([0.25, 0.25])))
        self.assertTrue(torch.equal(batch[6], torch.tensor([1, 1], dtype=torch.int8)))

    def test_policy_value_scoring_returns_legal_moves(self):
        model = ChessModel()
        model.value_head_trained = True
        board = chess.Board()

        scored = _get_move_scores(
            model, board, value_weight=1.0, value_candidate_limit=4
        )
        chosen = predict_next_move(
            model, board, temperature=0.0,
            value_weight=1.0, value_candidate_limit=4
        )

        self.assertEqual(len(scored), board.legal_moves.count())
        self.assertIn(chess.Move.from_uci(chosen), board.legal_moves)

    def test_terminal_value_is_from_previous_mover_perspective(self):
        board = chess.Board()
        for uci in ('f2f3', 'e7e5', 'g2g4', 'd8h4'):
            board.push(chess.Move.from_uci(uci))

        self.assertTrue(board.is_checkmate())
        self.assertEqual(_terminal_value_for_previous_mover(board), 1.0)


if __name__ == '__main__':
    unittest.main()
