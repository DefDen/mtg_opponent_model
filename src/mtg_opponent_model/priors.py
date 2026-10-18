"""Research priors: indexed historical decks and a smoothed generative mixture.

Requires the optional research dependencies. No test decklist is an input to
prediction; diagnostic use of recorded decks is separate and evaluation-only.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .data import DeckRecord
from .hands import PublicHistory, IncompatibleHistoryError

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


@dataclass(frozen=True)
class CardInfo:
    colors: str = ""
    types: str = ""
    mana_value: float = 0

    @property
    def is_land(self) -> bool:
        return "Land" in self.types


def load_card_info(path: Path) -> dict[str, CardInfo]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["name"]: CardInfo(row["color_identity"], row["types"],
                                    float(row["mana_value"] or 0)) for row in csv.DictReader(stream)}


@dataclass(frozen=True)
class Evidence:
    counts: Counter[str]
    hand_size: int
    required_size: int


def history_evidence(history: PublicHistory) -> Evidence:
    """Validate public accounting without consulting private hand/deck labels."""
    counts: Counter[str] = Counter()
    size = history.opening_hand_size
    if size < 0:
        raise ValueError("Negative opening size")
    unidentified = 0
    required = size
    for step in history.steps:
        if size + step.draws - len(step.departed) - step.unknown_departures != step.hand_size:
            raise ValueError("Inconsistent public hand accounting")
        required = max(required, sum(counts.values()) + unidentified + size + step.draws)
        counts.update(step.departed)
        unidentified += step.unknown_departures
        size = step.hand_size
    return Evidence(counts, size, required)


class IndexedDeckPrior:
    """Same random-departure prior as HandModel, with vectorized inference."""

    def __init__(self, decks: Iterable[DeckRecord], universe: Iterable[str]) -> None:
        self.decks = list(decks)
        self.cards = tuple(sorted(set(universe)))
        self.index = {card: i for i, card in enumerate(self.cards)}
        if not self.decks or not self.cards:
            raise ValueError("Need decks and a card universe")
        if any(set(d.cards) - self.index.keys() for d in self.decks):
            raise ValueError("Training deck contains an out-of-pool card")
        self.matrix = np.array([[d.cards.get(c, 0) for c in self.cards] for d in self.decks], dtype=np.int32)
        self.sizes = self.matrix.sum(axis=1)
        self.maximum = self.matrix.max(axis=0)

    def candidates(self, evidence: Evidence) -> np.ndarray:
        mask = self.sizes >= evidence.required_size
        for card, count in evidence.counts.items():
            if card not in self.index:
                return np.array([], dtype=int)
            mask &= self.matrix[:, self.index[card]] >= count
        return np.flatnonzero(mask)

    def predict(self, history: PublicHistory) -> dict[str, float]:
        evidence = history_evidence(history)
        selected = self.candidates(evidence)
        if not len(selected):
            raise IncompatibleHistoryError("No historical deck is compatible")
        remaining = self.matrix[selected].astype(float)
        sizes = self.sizes[selected].astype(float)
        log_weights = np.zeros(len(selected))
        removed = 0
        for card, count in evidence.counts.items():
            index = self.index[card]
            for i in range(count):
                log_weights += np.log(remaining[:, index] - i) - np.log(sizes - removed)
                removed += 1
            remaining[:, index] -= count
        totals = sizes - removed
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        absent = np.ones_like(remaining)
        for i in range(evidence.hand_size):
            absent *= np.maximum(totals[:, None] - remaining - i, 0) / (totals[:, None] - i)
        probabilities = np.clip(weights @ (1 - absent), 0, 1)
        return dict(zip(self.cards, probabilities.tolist()))

    def diagnose(self, history: PublicHistory, recorded_deck: Mapping[str, int]) -> dict:
        """Evaluation ONLY: separate finite-prior failures from pool violations.

        Exceeding the recorded deck suggests unmodeled reuse/generated cards or
        accounting errors; it does not prove the game itself was invalid.
        """
        evidence = history_evidence(history)
        selected = self.candidates(evidence)
        violations = {c: {"observed": n, "recorded": recorded_deck.get(c, 0)}
                      for c, n in evidence.counts.items() if n > recorded_deck.get(c, 0)}
        size_violation = evidence.required_size > sum(recorded_deck.values())
        if len(selected):
            reason = "supported"
        elif violations or size_violation:
            reason = "exceeds_recorded_single_deck_accounting"
        elif any(c not in self.index or self.maximum[self.index[c]] == 0 for c in evidence.counts):
            reason = "card_absent_from_training"
        elif evidence.required_size > self.sizes.max():
            reason = "deck_size_tail"
        elif any(n > self.maximum[self.index[c]] for c, n in evidence.counts.items()):
            reason = "copy_count_tail"
        else:
            reason = "unseen_card_combination"
        return {"reason": reason, "compatible_decks": len(selected),
                "recorded_pool_violation_evaluation_only": bool(violations or size_violation),
                "copy_violations_evaluation_only": violations,
                "size_violation_evaluation_only": size_violation}


def _log_rising(value: float, count: int) -> float:
    return sum(math.log(value + i) for i in range(count))


def _log_falling(value: int, count: int) -> float:
    if count > value:
        return -math.inf
    return sum(math.log(value - i) for i in range(count))


class SmoothedDeckPrior:
    """Color-group / deck-shape / Dirichlet-multinomial deck-count mixture.

    Draw a training-derived (deck size, land count) shape and color group, then
    fill its land and spell slots from separate smoothed Dirichlet categorical
    distributions. Every generated deck has exactly the selected size and land
    count. Limited has no general four-copy cap. All copies are bounded by the
    finite slot counts, and public departures consume slots without replacement.

    Exact marginal inference integrates new card combinations; it never edits
    public movements to make a history fit. Random-play, uniform draw, and
    unidentified-departure assumptions are the same as the historical baseline.
    """

    def __init__(self, decks: Iterable[DeckRecord], universe: Iterable[str],
                 metadata: Mapping[str, CardInfo], concentration: float = 50,
                 shrinkage: float = 50, floor: float = 0.01) -> None:
        decks = list(decks)
        self.cards = tuple(sorted(set(universe)))
        self.index = {c: i for i, c in enumerate(self.cards)}
        self.land = np.array([metadata.get(c, CardInfo()).is_land for c in self.cards])
        if not decks or not self.cards or min(concentration, shrinkage, floor) <= 0:
            raise ValueError("Need decks, a pool, and positive smoothing parameters")
        if any(set(d.cards) - self.index.keys() for d in decks):
            raise ValueError("Training deck contains an out-of-pool card")
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        shapes: Counter[tuple[str, int, int]] = Counter()
        global_counts = Counter(c for d in decks for c, n in d.cards.items() for _ in range(n))
        for deck in decks:
            colors: Counter[str] = Counter()
            for card, copies in deck.cards.items():
                for color in metadata.get(card, CardInfo()).colors:
                    if color in "WUBRG":
                        colors[color] += copies
            group = "".join(sorted(c for c, _ in sorted(colors.items(), key=lambda x: (-x[1], x[0]))[:2])) or "C"
            grouped[group].update(deck.cards)
            size = sum(deck.cards.values())
            lands = sum(n for c, n in deck.cards.items() if metadata.get(c, CardInfo()).is_land)
            shapes[group, size, lands] += 1
        self.alpha = {}
        for group, counts in grouped.items():
            alpha = np.zeros(len(self.cards))
            for mask in (self.land, ~self.land):
                indices = np.flatnonzero(mask)
                if not len(indices):
                    continue
                base = np.array([global_counts[self.cards[i]] + floor for i in indices], dtype=float)
                base /= base.sum()
                group_counts = np.array([counts[self.cards[i]] for i in indices], dtype=float)
                distribution = (group_counts + shrinkage * base) / (group_counts.sum() + shrinkage)
                alpha[indices] = concentration * distribution
            self.alpha[group] = alpha
        self.components = [(g, n, l, count / len(decks)) for (g, n, l), count in sorted(shapes.items())]
        self.concentration = concentration
        self.shrinkage = shrinkage
        self.floor = floor

    def supports(self, history: PublicHistory) -> bool:
        evidence = history_evidence(history)
        if set(evidence.counts) - self.index.keys():
            return False
        lands = sum(n for c, n in evidence.counts.items() if self.land[self.index[c]])
        spells = sum(evidence.counts.values()) - lands
        return any(size >= evidence.required_size and lands <= l and spells <= size-l
                   for _, size, l, _ in self.components)

    def predict(self, history: PublicHistory) -> dict[str, float]:
        evidence = history_evidence(history)
        if set(evidence.counts) - self.index.keys():
            raise IncompatibleHistoryError("Public card outside legal pool")
        seen = np.array([evidence.counts[c] for c in self.cards], dtype=int)
        kl = int(seen[self.land].sum())
        ks = int(seen[~self.land].sum())
        k = kl + ks
        group_terms = {}
        for group, alpha in self.alpha.items():
            term = 0.0
            for mask, observed in ((self.land, kl), (~self.land, ks)):
                if observed:
                    term += sum(_log_rising(float(alpha[i]), int(seen[i]))
                                for i in np.flatnonzero(mask & (seen > 0)))
                    term -= _log_rising(float(alpha[mask].sum()), observed)
            group_terms[group] = term
        supported = []
        for group, size, lands, prior in self.components:
            if size < evidence.required_size or kl > lands or ks > size - lands:
                continue
            weight = (math.log(prior) + _log_falling(lands, kl)
                      + _log_falling(size - lands, ks) - _log_falling(size, k)
                      + group_terms[group])
            supported.append((group, lands - kl, size - lands - ks, weight))
        if not supported:
            raise IncompatibleHistoryError("No finite deck shape can explain public movements")
        maximum = max(c[3] for c in supported)
        weights = np.array([math.exp(c[3] - maximum) for c in supported])
        weights /= weights.sum()
        result = np.zeros(len(self.cards))
        n = evidence.hand_size
        if n == 0:
            return dict.fromkeys(self.cards, 0.0)
        for (group, lands, spells, _), weight in zip(supported, weights):
            total = lands + spells
            denominator = math.comb(total, n)
            for mask, available, other in ((self.land, lands, spells), (~self.land, spells, lands)):
                if not available or not mask.any():
                    continue
                alpha = self.alpha[group][mask] + seen[mask]
                a_total = alpha.sum()
                absent = np.ones(len(alpha))
                presence = np.zeros(len(alpha))
                for m in range(min(n, available) + 1):
                    if m:
                        absent *= (a_total - alpha + m - 1) / (a_total + m - 1)
                    if n - m <= other:
                        mass = math.comb(available, m) * math.comb(other, n - m) / denominator
                        presence += mass * (1 - absent)
                result[mask] += weight * presence
        return dict(zip(self.cards, np.clip(result, 0, 1).tolist()))
