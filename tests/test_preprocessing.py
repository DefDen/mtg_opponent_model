import pytest

from mtg_opponent_model.data import DeckRecord
from mtg_opponent_model.preprocessing import CardVocabulary, split_decks


def test_stable_ids_counts_and_round_trip(tmp_path):
    vocab = CardVocabulary(["B", "A", "B"])
    assert vocab.card_to_id == {"A": 0, "B": 1}
    deck = DeckRecord("d", "g", {"A": 3, "unknown": 1})
    assert vocab.encode(deck) == {0: 3}
    path = tmp_path / "cards.json"
    vocab.save(path)
    assert CardVocabulary.load(path).cards == vocab.cards


def test_event_split_is_disjoint_and_reproducible():
    decks = [DeckRecord(str(i), str(i // 2), {"A": 1}) for i in range(8)]
    train, test = split_decks(decks, seed=42)
    assert train and test
    assert {d.group_id for d in train}.isdisjoint(d.group_id for d in test)
    assert (train, test) == split_decks(decks, seed=42)
    assert len(train) + len(test) == len(decks)


@pytest.mark.parametrize("cards", [{}, {"A": 0}, {"A": -1}, {"A": True}, {"": 1}])
def test_invalid_decks(cards):
    with pytest.raises(ValueError):
        DeckRecord("d", "g", cards)
