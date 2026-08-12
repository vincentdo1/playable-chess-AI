"""Fast checks for the reproducible Stockfish evaluation schedule."""

from types import SimpleNamespace

from evaluation.vs_stockfish import _schedule, _score_ci


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
