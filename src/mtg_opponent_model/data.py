"""Canonical records; source-specific adapters implement DeckLoader."""
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True)
class DeckRecord:
    """One deck configuration. group_id identifies its draft/event."""

    deck_id: str
    group_id: str
    cards: Mapping[str, int]
    expansion: str | None = None
    format: str | None = None

    def __post_init__(self) -> None:
        if not self.deck_id or not self.group_id or not self.cards:
            raise ValueError("Deck IDs, group IDs, and cards must be nonempty")
        if any(not card or type(count) is not int or count <= 0
               for card, count in self.cards.items()):
            raise ValueError("Cards require nonempty keys and positive integer counts")
        object.__setattr__(self, "cards", dict(self.cards))


class DeckLoader(Protocol):
    """Load complete main decks, preserving event grouping and copy counts."""

    def load(self) -> Iterable[DeckRecord]: ...
