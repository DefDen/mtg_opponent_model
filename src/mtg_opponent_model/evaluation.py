"""Seeded partial observations and macro-averaged retrieval metrics."""
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .data import DeckRecord
from .model import CooccurrenceModel
from .preprocessing import CardVocabulary


@dataclass(frozen=True)
class EvaluationCase:
    observed: frozenset[str]
    targets: frozenset[str]

    def __post_init__(self) -> None:
        if not self.targets or self.observed & self.targets:
            raise ValueError("Targets must be nonempty and disjoint from observations")


def sample_cases(
    decks: Iterable[DeckRecord], observed_count: int = 2, seed: int = 0,
) -> list[EvaluationCase]:
    """One case per eligible deck; skip decks with no hidden target."""
    if observed_count < 0:
        raise ValueError("observed_count must be nonnegative")
    rng = random.Random(seed)
    cases = []
    for deck in decks:
        cards = sorted(deck.cards)
        if len(cards) <= observed_count:
            continue
        observed = frozenset(rng.sample(cards, observed_count))
        cases.append(EvaluationCase(observed, frozenset(cards) - observed))
    return cases


def ranking_metrics(
    ranked: Sequence[str], targets: frozenset[str], ks: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    if not targets or any(k <= 0 for k in ks):
        raise ValueError("Need nonempty targets and positive cutoffs")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Ranking must contain unique cards")
    result = {f"Recall@{k}": len(set(ranked[:k]) & targets) / len(targets) for k in ks}
    result["MRR"] = next((1 / i for i, card in enumerate(ranked, 1)
                          if card in targets), 0.0)
    return result


def evaluate(
    model: CooccurrenceModel, vocabulary: CardVocabulary,
    cases: Iterable[EvaluationCase], ks: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    """Unknown targets remain in denominators; coverage is target micro-average."""
    results = []
    known_targets = total_targets = 0
    for case in cases:
        observed_ids = [vocabulary.card_to_id[c] for c in case.observed
                        if c in vocabulary.card_to_id]
        ranked = [vocabulary.cards[c] for c, _ in model.rank(observed_ids)]
        results.append(ranking_metrics(ranked, case.targets, ks))
        known_targets += len(case.targets & vocabulary.card_to_id.keys())
        total_targets += len(case.targets)
    if not results:
        raise ValueError("Need at least one evaluation case")
    metrics = {key: sum(row[key] for row in results) / len(results) for key in results[0]}
    metrics["target_coverage"] = known_targets / total_targets
    return metrics
