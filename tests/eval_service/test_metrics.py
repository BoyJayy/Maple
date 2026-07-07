from metrics import mrr_at_k, ndcg_at_k, recall_at_k, score


def test_recall_counts_hits_within_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, k=3) == 2 / 3
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == 1.0


def test_ndcg_penalizes_late_hits():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, k=10)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=10)
    assert early == 1.0
    assert 0 < late < early


def test_mrr_uses_first_hit_position():
    assert mrr_at_k(["x", "a", "b"], {"a", "b"}, k=10) == 0.5
    assert mrr_at_k(["a"], {"a"}, k=10) == 1.0
    assert mrr_at_k(["x", "y"], {"a"}, k=10) == 0.0
    assert mrr_at_k(["x", "a"], {"a"}, k=1) == 0.0


def test_score_weights_recall_over_ndcg():
    assert score(1.0, 0.0) == 0.8
    assert score(0.0, 1.0) == 0.2
