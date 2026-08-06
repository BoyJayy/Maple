from metrics import bootstrap_ci


def test_bootstrap_ci_empty_and_single():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5
    low, high = bootstrap_ci(values, seed=0)
    assert low <= sum(values) / len(values) <= high
    assert 0.0 <= low < high <= 1.0
    assert bootstrap_ci(values, seed=0) == (low, high)


def test_bootstrap_ci_narrows_with_constant_values():
    low, high = bootstrap_ci([0.7] * 50)
    assert low == high == 0.7
