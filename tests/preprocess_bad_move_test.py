import unittest

import chess
import preprocess
from training.preprocess import (
    CP_LOSS_BUCKET_BLUNDER,
    CP_LOSS_BUCKET_CRITICAL,
    CP_LOSS_BUCKET_INACCURACY,
    CP_LOSS_BUCKET_MISTAKE,
    CP_LOSS_BUCKET_OK,
    CP_LOSS_BUCKET_UNKNOWN,
    UNKNOWN_CP_LOSS,
    _cp_loss_bucket,
    _is_bad_move_from_cp_loss,
    _search_assisted_negative_moves,
    _select_search_negative_candidates,
)


class PreprocessBadMoveTest(unittest.TestCase):
    def test_cp_loss_bucket_thresholds(self):
        self.assertEqual(_cp_loss_bucket(UNKNOWN_CP_LOSS), CP_LOSS_BUCKET_UNKNOWN)
        self.assertEqual(_cp_loss_bucket(0), CP_LOSS_BUCKET_OK)
        self.assertEqual(_cp_loss_bucket(50), CP_LOSS_BUCKET_INACCURACY)
        self.assertEqual(_cp_loss_bucket(150), CP_LOSS_BUCKET_MISTAKE)
        self.assertEqual(_cp_loss_bucket(300), CP_LOSS_BUCKET_BLUNDER)
        self.assertEqual(_cp_loss_bucket(900), CP_LOSS_BUCKET_CRITICAL)

    def test_bad_move_flag_uses_training_threshold(self):
        self.assertFalse(_is_bad_move_from_cp_loss(UNKNOWN_CP_LOSS))
        self.assertFalse(_is_bad_move_from_cp_loss(149))
        self.assertTrue(_is_bad_move_from_cp_loss(150))

    def test_search_negative_candidate_selection_prioritizes_hanging_piece(self):
        board = chess.Board('3rk3/8/8/8/8/8/8/3QK3 w - - 0 1')
        played_move = chess.Move.from_uci('d1e2')
        bad_move = chess.Move.from_uci('d1d8')

        candidates = _select_search_negative_candidates(
            board, played_move, max_candidates=3
        )

        self.assertIn(bad_move, candidates)

    def test_search_assisted_negatives_label_clear_tactical_loss(self):
        board = chess.Board('3rk3/8/8/8/8/8/8/3QK3 w - - 0 1')
        played_move = chess.Move.from_uci('d1e2')
        bad_move = chess.Move.from_uci('d1d8')

        old_candidates = preprocess.SEARCH_NEGATIVE_CANDIDATES
        old_max = preprocess.SEARCH_NEGATIVE_MAX_PER_POSITION
        old_depth = preprocess.SEARCH_NEGATIVE_DEPTH
        old_threshold = preprocess.SEARCH_NEGATIVE_THRESHOLD
        try:
            preprocess.SEARCH_NEGATIVE_CANDIDATES = 5
            preprocess.SEARCH_NEGATIVE_MAX_PER_POSITION = 2
            preprocess.SEARCH_NEGATIVE_DEPTH = 2
            preprocess.SEARCH_NEGATIVE_THRESHOLD = 250

            negatives = dict(_search_assisted_negative_moves(board, played_move))

            self.assertIn(bad_move, negatives)
            self.assertGreaterEqual(negatives[bad_move], 250)
        finally:
            preprocess.SEARCH_NEGATIVE_CANDIDATES = old_candidates
            preprocess.SEARCH_NEGATIVE_MAX_PER_POSITION = old_max
            preprocess.SEARCH_NEGATIVE_DEPTH = old_depth
            preprocess.SEARCH_NEGATIVE_THRESHOLD = old_threshold


if __name__ == '__main__':
    unittest.main()
