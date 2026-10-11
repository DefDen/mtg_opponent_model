"""Stable card identities and leakage-safe event splitting."""
import json
import random
from pathlib import Path
from typing import Iterable

from .data import DeckRecord


class CardVocabulary:
    """Sorted card keys receive contiguous IDs; reuse/save this vocabulary."""

    def __init__(self, card_keys: Iterable[str]) -> None:
        self.cards = tuple(sorted(set(card_keys)))
        self.card_to_id = {card: i for i, card in enumerate(self.cards)}

    def encode(self, deck: DeckRecord) -> dict[int, int]:
        """Encode known cards; retain unknown keys separately for evaluation."""
        return {self.card_to_id[c]: n for c, n in deck.cards.items()
                if c in self.card_to_id}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.cards), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CardVocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def split_decks(
    decks: Iterable[DeckRecord], test_fraction: float = 0.2, seed: int = 0,
) -> tuple[list[DeckRecord], list[DeckRecord]]:
    """Keep all configurations of each event in the same partition."""
    records = list(decks)
    groups = sorted({d.group_id for d in records})
    if not 0 < test_fraction < 1 or len(groups) < 2:
        raise ValueError("Need a fraction in (0, 1) and at least two events")
    random.Random(seed).shuffle(groups)
    size = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    test_groups = set(groups[:size])
    return ([d for d in records if d.group_id not in test_groups],
            [d for d in records if d.group_id in test_groups])
