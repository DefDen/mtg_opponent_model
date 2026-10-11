import pytest

from mtg_opponent_model.data import DeckRecord
from mtg_opponent_model.hands import HandModel, PublicHistory, PublicStep, hand_metrics


def test_exact_copy_aware_hand_probabilities():
    model = HandModel([DeckRecord("d", "g", {"A": 2, "B": 2})])
    assert model.predict(PublicHistory(2, ())) == pytest.approx({"A": 5 / 6, "B": 5 / 6})
    history = PublicHistory(2, (PublicStep(1, "target", 0, ("A",), 1),))
    assert model.predict(history) == pytest.approx({"A": 1 / 3, "B": 2 / 3})


def test_public_identity_updates_deck_weights():
    model = HandModel([DeckRecord("a", "a", {"A": 2, "B": 2}),
                       DeckRecord("b", "b", {"A": 1, "C": 3})])
    history = PublicHistory(2, (PublicStep(1, "target", 0, ("A",), 1),))
    # Posterior deck weights 2/3 and 1/3 after observing A leave hand.
    assert model.predict(history) == pytest.approx({"A": 2 / 9, "B": 4 / 9, "C": 1 / 3})


def test_played_card_can_still_be_held_and_draws_are_anonymous():
    model = HandModel([DeckRecord("d", "g", {"A": 2, "B": 2})])
    history = PublicHistory(2, (PublicStep(1, "target", 1, ("A",), 2),))
    assert model.predict(history)["A"] == pytest.approx(2 / 3)
    exhausted = PublicHistory(2, (PublicStep(1, "target", 0, ("A", "A"), 0),))
    assert model.predict(exhausted) == {"A": 0, "B": 0}


def test_incompatible_or_inconsistent_history_raises():
    model = HandModel([DeckRecord("d", "g", {"A": 2})])
    with pytest.raises(ValueError, match="compatible"):
        model.predict(PublicHistory(1, (PublicStep(1, "target", 0, ("B",), 0),)))
    with pytest.raises(ValueError, match="Hand size"):
        model.predict(PublicHistory(1, (PublicStep(1, "target", 1, (), 1),)))


def test_brier_and_unknown_targets():
    result = hand_metrics({"A": 0.75, "B": 0.25}, {"A": 1, "unseen": 1}, ["A", "B", "unseen"])
    assert result["Recall@5"] == 0.5
    assert result["MRR"] == 1
    assert result["Brier"] == pytest.approx((0.25 ** 2 + 0.25 ** 2 + 1) / 3)
    empty = hand_metrics({"A": 0}, {}, ["A"])
    assert empty["Brier"] == 0
    assert empty["MRR"] == 0
