"""Deck-presence conditional frequencies, without learned parameters."""
from collections import Counter
from typing import Iterable


class CooccurrenceModel:
    """Mean P(candidate | observed card); scores need not sum to one."""

    def __init__(self) -> None:
        self.frequency: Counter[int] = Counter()
        self.pairs: Counter[tuple[int, int]] = Counter()
        self.deck_count = 0

    def fit(self, decks: Iterable[Iterable[int]]) -> "CooccurrenceModel":
        frequency: Counter[int] = Counter()
        pairs: Counter[tuple[int, int]] = Counter()
        count = 0
        for deck in decks:
            cards = set(deck)
            if any(type(c) is not int or c < 0 for c in cards):
                raise ValueError("Card IDs must be nonnegative integers")
            if not cards:
                continue
            count += 1
            frequency.update(cards)
            pairs.update((a, b) for a in cards for b in cards if a != b)
        if not count:
            raise ValueError("Need at least one nonempty training deck")
        self.frequency, self.pairs, self.deck_count = frequency, pairs, count
        return self

    def rank(self, observed: Iterable[int], k: int | None = None) -> list[tuple[int, float]]:
        if not self.deck_count:
            raise RuntimeError("Fit the model before ranking")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        seen = set(observed)
        known = seen & self.frequency.keys()
        scores = []
        for card in self.frequency.keys() - seen:
            score = (sum(self.pairs[o, card] / self.frequency[o] for o in known)
                     / len(known) if known else self.frequency[card] / self.deck_count)
            scores.append((card, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores if k is None else scores[:k]
