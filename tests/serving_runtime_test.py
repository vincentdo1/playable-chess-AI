"""Focused regression tests for production serving controls."""

from types import SimpleNamespace

import chess
import numpy as np

import backend.app as backend_app
import inference.mcts_player as mcts_player
import inference.search_player as search_player
from neural_network import (
    BOARD_ENCODING_VERSION,
    BOARD_ENCODING_VERSION_V3,
    get_encoding_spec,
)


class _SpecOnlyModel:
    def __init__(self):
        self.encoding_spec = get_encoding_spec(BOARD_ENCODING_VERSION)


def _fake_batched_eval(model, boards):
    spec = model.encoding_spec
    logits = np.zeros(
        (len(boards), spec['move_vocab_size']), dtype=np.float32,
    )
    root_move = chess.Move.from_uci('e2e4')
    logits[:, spec['move_to_index'](root_move, flip=False)] = 10.0
    values = np.zeros(len(boards), dtype=np.float32)
    return logits, values


def test_mcts_deadline_before_first_simulation_uses_policy_prior(monkeypatch):
    clock = [0.0]

    def slow_root_eval(model, boards):
        result = _fake_batched_eval(model, boards)
        clock[0] = 2.0
        return result

    monkeypatch.setattr(mcts_player, '_batched_eval', slow_root_eval)
    monkeypatch.setattr(mcts_player.time, 'monotonic', lambda: clock[0])

    move, stats = mcts_player.mcts_search_best_move_with_stats(
        _SpecOnlyModel(), chess.Board(), num_simulations=8, time_limit=1.0,
    )

    assert move == chess.Move.from_uci('e2e4')
    assert stats.simulations == 0
    assert stats.stop_reason == 'time_limit'


def test_mcts_tiny_temperature_keeps_zero_visit_prior_fallback_stable():
    root = mcts_player.MCTSNode()
    preferred = chess.Move.from_uci('e2e4')
    other = chess.Move.from_uci('d2d4')
    root.children[preferred] = mcts_player.MCTSNode(prior=0.9)
    root.children[other] = mcts_player.MCTSNode(prior=0.1)

    assert mcts_player.select_move_from_root(root, 1e-12) == preferred


def test_mcts_never_exceeds_simulation_budget(monkeypatch):
    monkeypatch.setattr(mcts_player, '_batched_eval', _fake_batched_eval)
    _, stats = mcts_player.mcts_search(
        _SpecOnlyModel(), chess.Board(), num_simulations=5, batch_size=4,
    )
    assert stats.simulations == 5
    assert stats.stop_reason == 'simulation_limit'


def test_search_cache_key_includes_v3_clock_and_v2_history():
    start = chess.Board()
    same_position_later = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        same_position_later.push_uci(uci)

    assert (
        chess.polyglot.zobrist_hash(start)
        == chess.polyglot.zobrist_hash(same_position_later)
    )

    evaluator = search_player.NNEvaluator.__new__(search_player.NNEvaluator)
    evaluator.encoding_version = BOARD_ENCODING_VERSION_V3
    assert evaluator.cache_key(start) != evaluator.cache_key(same_position_later)

    evaluator.encoding_version = BOARD_ENCODING_VERSION
    assert evaluator.cache_key(start) != evaluator.cache_key(same_position_later)


def test_timed_out_search_restores_caller_board(monkeypatch):
    class FakeEvaluator:
        def __init__(self, _model, stats=None):
            self.stats = stats

        def cache_key(self, board):
            return (chess.polyglot.zobrist_hash(board),)

    monkeypatch.setattr(search_player, 'NNEvaluator', FakeEvaluator)
    monkeypatch.setattr(
        search_player, '_ordered_moves',
        lambda _evaluator, board, _hash_move: list(board.legal_moves)[:1],
    )
    monkeypatch.setattr(
        search_player, '_negamax',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    board = chess.Board()
    original_fen = board.fen()
    search_player.search_best_move(object(), board, max_depth=1)
    assert board.fen() == original_fen
    assert not board.move_stack


def _install_fake_policy(monkeypatch):
    monkeypatch.setattr(backend_app, '_magnus_model', object())
    monkeypatch.setattr(
        backend_app, '_predict_fn', lambda _model, _board, **_kwargs: 'e2e4',
    )


def test_request_booleans_are_strict(monkeypatch):
    _install_fake_policy(monkeypatch)
    response, status = backend_app._get_magnus_move(
        chess.Board(), {'use_mcts': 'false'},
    )
    assert status == 400
    assert response['error'] == "'use_mcts' must be a boolean"


def test_client_cannot_enable_server_disabled_mcts(monkeypatch):
    _install_fake_policy(monkeypatch)
    monkeypatch.setattr(backend_app, 'DEFAULT_MAGNUS_USE_MCTS', False)
    monkeypatch.setattr(
        backend_app, '_mcts_fn',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    response, status = backend_app._get_magnus_move(
        chess.Board(), {'use_mcts': True, 'blunder_guard': False},
    )
    assert status == 200
    assert response['method'] == 'policy'


def test_mcts_request_can_only_reduce_server_budget(monkeypatch):
    _install_fake_policy(monkeypatch)
    captured = {}

    def fake_mcts(_model, _board, **kwargs):
        captured.update(kwargs)
        return chess.Move.from_uci('e2e4'), SimpleNamespace(
            simulations=48,
            elapsed=0.25,
            stop_reason='time_limit',
        )

    monkeypatch.setattr(backend_app, 'DEFAULT_MAGNUS_USE_MCTS', True)
    monkeypatch.setattr(backend_app, 'MAGNUS_MCTS_SIMULATION_LIMIT', 64)
    monkeypatch.setattr(backend_app, '_mcts_fn', fake_mcts)

    response, status = backend_app._get_magnus_move(
        chess.Board(), {'use_mcts': True, 'mcts_simulations': 500},
    )
    assert status == 200
    assert captured['num_simulations'] == 64
    assert response['mcts_simulations_requested'] == 500
    assert response['mcts_simulations_budget'] == 64
    assert response['mcts_simulations'] == 48


def test_required_model_controls_readiness_but_not_liveness(monkeypatch):
    monkeypatch.setattr(backend_app, 'MAGNUS_REQUIRED', True)
    monkeypatch.setattr(backend_app, '_magnus_model', None)
    monkeypatch.setattr(backend_app, '_predict_fn', None)

    client = backend_app.app.test_client()
    assert client.get('/readyz').status_code == 503
    assert client.get('/').status_code == 503
    assert client.get('/livez').status_code == 200


def test_busy_server_rejects_work_without_waiting(monkeypatch):
    class BusySlots:
        def acquire(self, blocking=False):
            assert blocking is False
            return False

    monkeypatch.setattr(backend_app, '_inference_slots', BusySlots())
    response = backend_app.app.test_client().post('/api/move', json={
        'fen': chess.STARTING_FEN,
        'player': 'alphabeta',
    })
    assert response.status_code == 429
    assert response.headers['Retry-After'] == '1'


def test_oversized_request_is_rejected_before_json_parsing():
    response = backend_app.app.test_client().post(
        '/api/move',
        data=b'{' + (b' ' * backend_app.MAX_REQUEST_BYTES) + b'}',
        content_type='application/json',
    )
    assert response.status_code == 413


def test_non_finite_request_value_is_rejected(monkeypatch):
    _install_fake_policy(monkeypatch)
    response, status = backend_app._get_magnus_move(
        chess.Board(), {'temperature': float('nan')},
    )
    assert status == 400
    assert response['error'] == "'temperature' must be finite"
