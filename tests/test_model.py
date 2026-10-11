import pytest

from mtg_opponent_model.model import CooccurrenceModel


def test_conditional_frequency_and_duplicate_presence():
    model = CooccurrenceModel().fit([[0, 0, 1], [0, 2], [0, 1, 2], [3]])
    assert model.rank([0, 0]) == [(1, 2 / 3), (2, 2 / 3), (3, 0.0)]
    assert model.rank([0, 1]) == [(2, (2 / 3 + 1 / 2) / 2), (3, 0.0)]
    assert model.rank([0], k=1) == [(1, 2 / 3)]
    assert model.rank([0], k=0) == []


def test_fallback_and_refit():
    model = CooccurrenceModel().fit([[0, 1], [0, 2]])
    assert model.rank([]) == [(0, 1.0), (1, 0.5), (2, 0.5)]
    assert model.rank([999]) == model.rank([])
    assert model.rank([0, 999]) == model.rank([0])
    model.fit([[3]])
    assert model.rank([]) == [(3, 1.0)]


def test_invalid_model_usage():
    with pytest.raises(RuntimeError):
        CooccurrenceModel().rank([])
    with pytest.raises(ValueError):
        CooccurrenceModel().fit([])
    with pytest.raises(ValueError):
        CooccurrenceModel().fit([[-1]])
    with pytest.raises(ValueError):
        CooccurrenceModel().fit([[0]]).rank([], -1)
