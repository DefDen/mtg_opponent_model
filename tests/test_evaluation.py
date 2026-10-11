import pytest

from mtg_opponent_model.data import DeckRecord
from mtg_opponent_model.evaluation import EvaluationCase, evaluate, ranking_metrics, sample_cases
from mtg_opponent_model.model import CooccurrenceModel
from mtg_opponent_model.preprocessing import CardVocabulary, split_decks


def test_metrics_use_full_target_denominator_and_first_hit():
    ranked = ["miss", "A", "B"]
    scores = ranking_metrics(ranked, frozenset({"A", "B", "unseen"}), (1, 2, 5))
    assert scores == {"Recall@1": 0, "Recall@2": 1 / 3, "Recall@5": 2 / 3, "MRR": 0.5}
    assert ranking_metrics([], frozenset({"A"}))["MRR"] == 0


def test_seeded_sampling():
    decks = [DeckRecord("d", "g", {"A": 2, "B": 1, "C": 1})]
    cases = sample_cases(decks, 1, 17)
    assert cases == sample_cases(decks, 1, 17)
    assert len(cases[0].observed) == 1
    assert cases[0].observed | cases[0].targets == frozenset(decks[0].cards)
    assert sample_cases(decks, 3) == []


def test_macro_average_and_unknown_targets():
    vocab = CardVocabulary(["A", "B"])
    model = CooccurrenceModel().fit([[0, 1]])
    cases = [EvaluationCase(frozenset({"A"}), frozenset({"B", "unknown"})),
             EvaluationCase(frozenset(), frozenset({"A"}))]
    scores = evaluate(model, vocab, cases)
    assert scores == {"Recall@5": 0.75, "Recall@10": 0.75, "Recall@20": 0.75,
                      "MRR": 1.0, "target_coverage": 2 / 3}


def test_training_statistics_exclude_held_out_events():
    decks = [DeckRecord(str(i), str(i), {f"card{i}": 1, "shared": 1}) for i in range(5)]
    train, test = split_decks(decks)
    vocab = CardVocabulary(c for deck in train for c in deck.cards)
    model = CooccurrenceModel().fit(vocab.encode(deck) for deck in train)
    assert all(f"card{deck.deck_id}" not in vocab.card_to_id for deck in test)
    scores = evaluate(model, vocab, sample_cases(test, observed_count=0))
    assert scores["target_coverage"] == 0.5
    assert scores["Recall@20"] == 0.5


def test_invalid_evaluation():
    with pytest.raises(ValueError):
        ranking_metrics(["A", "A"], frozenset({"A"}))
    with pytest.raises(ValueError):
        EvaluationCase(frozenset({"A"}), frozenset({"A"}))
    with pytest.raises(ValueError):
        evaluate(CooccurrenceModel().fit([[0]]), CardVocabulary(["A"]), [])
