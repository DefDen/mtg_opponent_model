"""Synthetic smoke demonstration, not a research performance estimate."""
from mtg_opponent_model.data import DeckRecord
from mtg_opponent_model.evaluation import evaluate, sample_cases
from mtg_opponent_model.model import CooccurrenceModel
from mtg_opponent_model.preprocessing import CardVocabulary, split_decks


def main() -> None:
    decks = [DeckRecord(str(i), str(i), cards) for i, cards in enumerate([
        {"Plains": 8, "Island": 9, "White creature": 2, "Blue creature": 2, "Counterspell": 1},
        {"Plains": 9, "Island": 8, "White creature": 3, "Blue creature": 1, "Counterspell": 2},
        {"Swamp": 9, "Mountain": 8, "Black creature": 2, "Red creature": 2, "Removal": 1},
        {"Swamp": 8, "Mountain": 9, "Black creature": 3, "Red creature": 1, "Removal": 2},
        {"Plains": 9, "Island": 8, "White creature": 2, "Blue creature": 2, "Counterspell": 2},
        {"Swamp": 9, "Mountain": 8, "Black creature": 2, "Red creature": 2, "Removal": 2},
    ])]
    train, test = split_decks(decks, test_fraction=1 / 3, seed=42)
    vocabulary = CardVocabulary(card for deck in train for card in deck.cards)
    model = CooccurrenceModel().fit(vocabulary.encode(deck) for deck in train)
    print("Synthetic demonstration (not historical-data results)")
    for metric, value in evaluate(model, vocabulary, sample_cases(test, 2, 42)).items():
        print(f"{metric}: {value:.3f}")
    observed = [vocabulary.card_to_id["Plains"]]
    print("Predictions given Plains:")
    for card, score in model.rank(observed, k=5):
        print(f"  {vocabulary.cards[card]}: {score:.3f}")


if __name__ == "__main__":
    main()
