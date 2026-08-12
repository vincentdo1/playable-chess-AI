"""Fast checks for the reproducible Stockfish evaluation schedule."""

from types import SimpleNamespace

from evaluation.vs_stockfish import _schedule, _score_ci, _t_critical_975


def test_paired_schedule_reverses_colors_per_opening():
    args = SimpleNamespace(
        openings=None,
        no_openings=False,
        games=8,
        paired=True,
        seed=17,
    )
    schedule = _schedule(args)
    assert len(schedule) == 8
    for index in range(0, len(schedule), 2):
        first, second = schedule[index:index + 2]
        assert first[:2] == second[:2]
        assert first[2] is True and second[2] is False


def test_score_interval_uses_pair_units():
    # Four color-reversed pairs, not eight falsely-independent games.
    pair_scores = [0.75, 0.50, 0.25, 0.50]
    low, high = _score_ci(pair_scores)
    assert low < 0.5 < high
    assert 0.0 <= low <= high <= 1.0


def test_score_interval_is_a_t_interval_not_normal():
    # 12 paired units (a 24-game gauntlet). A z-interval with 1.96 would be
    # narrower than the correct t-interval with 11 degrees of freedom.
    import math
    import statistics

    pair_scores = [1.0, 0.5, 0.5, 0.75, 0.25, 0.5, 1.0, 0.5, 0.75, 0.5, 0.25, 0.5]
    low, high = _score_ci(pair_scores)
    mean = statistics.fmean(pair_scores)
    se = statistics.stdev(pair_scores) / math.sqrt(len(pair_scores))
    assert math.isclose(low, mean - 2.201 * se, abs_tol=1e-9)
    assert low < mean - 1.96 * se  # strictly wider than the normal interval


def test_t_critical_is_monotone_and_conservative_between_anchors():
    assert _t_critical_975(15) == 2.131
    assert _t_critical_975(31) == 2.042   # falls back to the smaller-df value
    assert _t_critical_975(45) == 2.021
    assert _t_critical_975(1000) == 1.980
    values = [_t_critical_975(df) for df in range(1, 200)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert min(values) >= 1.96
