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

The source is [17Lands public datasets](https://www.17lands.com/public_datasets),
licensed CC BY 4.0 unless otherwise noted. Data attribution: 17Lands.
`SeventeenLandsLoader` in `seventeenlands.py` reads plain or gzip game CSVs.
It extracts `deck_` copy counts and groups by `draft_id`. It ignores sideboards
and drawn/hand columns, deduplicates identical configurations within each
draft, and skips decks with fewer than 40 cards. Malformed counts, missing
metadata, and unexpected sets/formats raise errors. Different configurations
remain separate records but stay together in the split; events with more
configurations therefore receive more weight. Raw data is ignored by Git.

Download and run the historical experiment:

```bash
mkdir -p data
curl --fail --location --output data/game_data_public.BLB.PremierDraft.csv.gz \
  https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.BLB.PremierDraft.csv.gz
.venv/bin/python examples/historical.py data/game_data_public.BLB.PremierDraft.csv.gz
```

The default reads the first 20,000 game rows, not a random sample. Use
`--max-rows 0` to read the full file. Other options include `--observed-count`,
`--seed`, `--expansion`, `--format`, and `--output`. The runner writes a JSON
report with source checksum, counts, settings, and metrics, plus a vocabulary
JSON alongside it. It compares co-occurrence with training-card popularity,
excluding observed cards in both models. Models are fitted anew on each run.

See [the historical smoke results](reports/blb_smoke.md) for the initial run.

Uniformly hiding cards tests deck reconstruction; it does not reproduce the
biased sequence of cards revealed during actual play. Action-conditioned
inference, remaining-copy prediction, sideboards, and replay integration are
later milestones. Pair counting is quadratic in each deck's distinct cards,
which is appropriate for this initial Limited baseline.
