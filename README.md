# MTG opponent modeling baseline

A small, interpretable baseline for reconstructing Limited deck membership
from observed cards. Python 3.10+; no runtime dependencies, network model, or UI.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python examples/baseline.py
```

The example uses synthetic decks only. Its scores are a smoke demonstration,
not evidence of performance on historical Limited data.

## Interfaces and workflow

- `data.py`: `DeckRecord` holds opaque card keys and positive copy counts,
  deck ID, event/group ID, and optional expansion/format. A `DeckLoader`
  implements `load() -> Iterable[DeckRecord]`; no external schema is assumed.
- `preprocessing.py`: split whole events first, then build `CardVocabulary`
  from training cards. Sorted keys receive unique integer IDs starting at zero.
  Save the vocabulary with model artifacts; IDs can change if rebuilt with a
  different card universe. Encoding preserves counts and omits unknown keys.
- `model.py`: fit on encoded deck keys. Rank candidates given observed IDs,
  returning `(card_id, score)` pairs, optionally limited to k results.
- `evaluation.py`: sample distinct observed cards from held-out decks, then
  evaluate the full ranking against remaining distinct cards.

Co-occurrence counts use presence once per deck, regardless of copies:

```
P(candidate | observed) = decks containing both / decks containing observed
score(candidate) = mean P(candidate | observed) over known observed cards
```

Observed cards are excluded. Unknown observations are ignored; if none are
known, ranking falls back to training deck prevalence. Ties use ascending ID.
A new fit replaces previous statistics. Scores are interpretable conditional
frequency averages, not a normalized distribution over hands or libraries.

Recall@5, Recall@10, and Recall@20 measure recovered targets / all hidden
targets. MRR is the reciprocal rank of the first target, or zero if absent.
Metrics are averaged equally across cases. Unknown targets remain in recall
denominators; `target_coverage` reports the fraction of all target occurrences
in the vocabulary. Decks with too few distinct cards for the requested
observation count are skipped; an empty evaluation raises an error. Use one
set and one format per experiment. Basic lands are included unless filtered
explicitly before fitting and evaluation.

## Historical data integration

The intended source is [17Lands public datasets](https://www.17lands.com/public_datasets).
No dataset has been downloaded and no historical results are claimed yet.
A future adapter will inspect an actual game-data file, extract complete player
main decks, and retain copy counts. Deduplicate repeated game rows by event
and deck configuration; retain event IDs so revisions cannot cross partitions.
Do not merge identical lists across unrelated events. The core should receive
canonical records rather than source-specific columns. Attribute 17Lands
according to the dataset license when publishing results.

Uniformly hiding cards tests deck reconstruction; it does not reproduce the
biased sequence of cards revealed during actual play. Action-conditioned
inference, remaining-copy prediction, sideboards, and replay integration are
later milestones. Pair counting is quadratic in each deck's distinct cards,
which is appropriate for this initial Limited baseline.
