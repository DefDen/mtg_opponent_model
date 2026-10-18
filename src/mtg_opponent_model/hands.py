"""Public card movements and an exact, exchangeable hand baseline."""
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from .data import DeckRecord


@dataclass(frozen=True)
class PublicStep:
    """One half-turn summary; card identities entering the hand are hidden."""

    turn: int
    active_player: str  # target or observer
    draws: int
    departed: tuple[str, ...]  # target lands, casts, and public discards
    hand_size: int
    unknown_departures: int = 0  # Public count; omitted identities stay latent.

    def __post_init__(self) -> None:
        if (self.turn < 1 or self.active_player not in {"target", "observer"}
                or self.draws < 0 or self.hand_size < 0
                or type(self.unknown_departures) is not int or self.unknown_departures < 0):
            raise ValueError("Invalid public step")


@dataclass(frozen=True)
class PublicHistory:
    """Model input: no opening hand, draw identities, decklist, or hand labels."""

    opening_hand_size: int
    steps: tuple[PublicStep, ...]


class IncompatibleHistoryError(ValueError):
    """The finite training-deck prior cannot explain a public history."""


class HandModel:
    """Mixture of historical decks under random draws and random departures.

    Every distinct deck configuration has equal prior weight. Departing card
    identities update deck weights without replacement. Given a compatible
    deck, the current hand is a uniform subset of its remaining cards. This is
    exact for the assumed random-departure policy, not a strategic play model.
    Tutors, returns, and casting from other zones are unsupported. A smaller
    opening size can approximate a kept mulligan hand, but keep decisions and
    London-mulligan bottoming order are not modeled.

    Unidentified departures are marginalized under the same random-selection
    assumption. Their cards remain in the latent pool, not necessarily in the
    library; the current hand is a uniform subset of that pool. Their counts
    still reduce the number of cards available for subsequent draws.
    """

    def __init__(self, decks: Iterable[DeckRecord]) -> None:
        self.decks = [dict(deck.cards) for deck in decks]
        self.cards = tuple(sorted({card for deck in self.decks for card in deck}))
        if not self.decks:
            raise ValueError("Need training decks")

    def predict(self, history: PublicHistory) -> dict[str, float]:
        """P(at least one copy currently held); marginals need not sum to one.

        Raise on unsupported public histories or if no prior deck is compatible.
        """
        size = history.opening_hand_size
        if size < 0:
            raise ValueError("Opening hand size must be nonnegative")
        departed: list[str] = []
        required_deck_size = size
        unidentified = 0
        for step in history.steps:
            if size + step.draws - len(step.departed) - step.unknown_departures != step.hand_size:
                raise ValueError("Hand size does not match public card movements")
            required_deck_size = max(required_deck_size,
                                     len(departed) + unidentified + size + step.draws)
            departed.extend(step.departed)
            unidentified += step.unknown_departures
            size = step.hand_size
        posterior: list[tuple[dict[str, int], float]] = []
        for original in self.decks:
            remaining = dict(original)
            total = sum(remaining.values())
            if total < required_deck_size:
                continue
            log_weight = 0.0
            for card in departed:
                copies = remaining.get(card, 0)
                if not copies:
                    break
                log_weight += math.log(copies / total)
                remaining[card] -= 1
                total -= 1
            else:
                if size <= total:
                    posterior.append((remaining, log_weight))
        if not posterior:
            raise IncompatibleHistoryError("No training deck is compatible with the public history")
        max_weight = max(weight for _, weight in posterior)
        weights = [math.exp(weight - max_weight) for _, weight in posterior]
        denominator = sum(weights)
        result = dict.fromkeys(self.cards, 0.0)
        for (remaining, _), weight in zip(posterior, weights):
            total = sum(remaining.values())
            for card, count in remaining.items():
                if not count or not size:
                    continue
                # Probability of no copies in a sample of `size` cards.
                absent = (math.comb(total - count, size) / math.comb(total, size)
                          if total - count >= size else 0.0)
                result[card] += weight / denominator * (1 - absent)
        return result


def hand_metrics(probabilities: Mapping[str, float], hand: Mapping[str, int],
                 universe: Iterable[str]) -> dict[str, float]:
    """Binary card-presence Brier score plus recall and first-hit MRR.

    Use a fixed experiment universe, including unknown target cards. Empty
    hands have zero recall/MRR and still contribute to the Brier score.
    """
    cards = sorted(set(universe))
    if not cards or not set(hand).issubset(cards) or not set(probabilities).issubset(cards):
        raise ValueError("Need a universe containing predictions and targets")
    if any(not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Probabilities must be in [0, 1]")
    targets = {c for c, count in hand.items() if count > 0}
    ranked = sorted((c for c in probabilities if probabilities[c] > 0),
                    key=lambda c: (-probabilities[c], c))
    result = {f"Recall@{k}": (len(set(ranked[:k]) & targets) / len(targets)
                               if targets else 0.0) for k in (5, 10, 20)}
    result["MRR"] = next((1 / i for i, c in enumerate(ranked, 1) if c in targets), 0.0)
    result["Brier"] = sum((probabilities.get(c, 0) - int(c in targets)) ** 2
                           for c in cards) / len(cards)
    return result
