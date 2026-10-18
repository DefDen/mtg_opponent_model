"""Ranking and calibration diagnostics on a fixed card universe."""
import math
from typing import Iterable, Mapping

import numpy as np

from .hands import hand_metrics


def probability_metrics(probabilities: Mapping[str, float], actual: Mapping[str, int],
                        universe: Iterable[str]) -> dict[str, float]:
    cards = tuple(sorted(universe))
    result = hand_metrics(probabilities, actual, cards)
    loss = 0.0
    for card in cards:
        p = max(1e-6, min(1-1e-6, probabilities.get(card, 0)))
        loss -= math.log(p if actual.get(card, 0) > 0 else 1-p)
    result['LogLoss'] = loss / len(cards)
    return result


def calibration_bins(predictions: list[Mapping[str, float]], actual: list[Mapping[str, int]],
                     cards: Iterable[str], bins: int = 10) -> dict:
    """Equal-weight card/snapshot reliability bins; all negatives are retained."""
    if bins < 1 or len(predictions) != len(actual):
        raise ValueError("Need positive bins and paired predictions/labels")
    counts = np.zeros(bins,dtype=int)
    sums = np.zeros(bins)
    positives = np.zeros(bins)
    cards = tuple(cards)
    for probs, hand in zip(predictions,actual):
        for card in cards:
            p = probs.get(card,0)
            b = min(bins-1, int(p*bins))
            counts[b]+=1; sums[b]+=p; positives[b]+=int(hand.get(card,0)>0)
    total = int(counts.sum())
    rows = [{'lower':i/bins,'upper':(i+1)/bins,'count':int(counts[i]),
             'mean_probability':float(sums[i]/counts[i]) if counts[i] else None,
             'observed_fraction':float(positives[i]/counts[i]) if counts[i] else None}
            for i in range(bins)]
    ece = sum(abs(sums[i]-positives[i]) for i in range(bins))/total if total else 0
    return {'expected_calibration_error':float(ece),'card_snapshot_pairs':total,'bins':rows}
