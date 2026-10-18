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
biased sequence of cards revealed during actual play. Pair counting is
quadratic in each deck's distinct cards, which is appropriate for this baseline.

## Hand prediction from replay history

The next baseline predicts **the probability of holding at least one copy of
each card right now**. It uses public draw counts, hand sizes, and cards played,
cast, or publicly discarded. It does not yet infer information from passing,
attacking, mana availability, or strategic card selection.

- `replays.py` normalizes Arena IDs with the 17Lands card mapping, reconstructs
  early-turn hand movements, and separates `HandSnapshot.actual_hand` labels
  from `PublicHistory` inputs. Draw identities and opening hands are private;
  only their lengths cross into public inputs. Replay records contain one game
  per wide CSV row, with separate columns for each half-turn.
- `hands.py` implements a mixture over historical decks, retaining copy counts.
  Public departures reweight compatible decks by their without-replacement
  likelihoods. Given each deck, the residual hand is assumed to be a uniform
  sample of the remaining cards; an exact hypergeometric calculation gives
  per-card holding probabilities. Marginals do not sum to one. Played cards
  can still be held if other copies remain.
- `examples/replay_hands.py` compares this model with the same deck prior using
  only current hand size. Held-out draft IDs are excluded from **all** prior
  decks. Evaluation averages snapshots within games, then averages games.
  Brier score measures binary card-presence probability error across a fixed
  schema-derived universe (lower is better). Zero-probability cards are excluded
  from ranking, and unknown targets remain in recall denominators.

Prepare a small replay sample and run:

```bash
.venv/bin/python examples/download_replay_sample.py
curl --fail --location --output data/cards.csv \
  https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv
.venv/bin/python examples/replay_hands.py --output reports/blb_hands_smoke.json
```

The downloader requests only the first 8 MiB of compressed replay data, then
writes 500 complete game rows to a valid gzip file. If a larger sample cannot
fit, increase `--compressed-bytes`. The experiment also needs the historical
game-data file downloaded above. The report includes public histories, top
predictions, actual hands marked evaluation-only, exclusion counts, and source
checksums. See [the hand experiment report](reports/blb_hands_smoke.md).

This first replay experiment uses the first three rounds and zero-mulligan
games only. It stops a game when recorded movements fail to reproduce the
exact hand, or when a tutor is recorded. Off-turn draws, returns, flash
creatures, and other unrepresented movements can therefore truncate games.
No-compatible-prior snapshots are excluded and counted. Consistent hand
accounting does not prove a cast came from hand: zone provenance and exact
within-turn event order are unavailable. The model ignores library changes
from milling, scrying, and other effects. Its random-departure policy and uniform
opening hands ignore player strategy and Arena hand smoothing. Filtering and
the file-prefix sample introduce selection bias; these are smoke results.

Future work: broaden reliable event reconstruction, handle mulligans and
library/zone effects, and learn action likelihoods before interpreting passes
and other choices as evidence about hidden cards.

To inspect reconstruction failures across the existing sample, run
`.venv/bin/python examples/audit_replays.py`. It audits rounds 3, 10, and 30,
counts stops by turn and player, and writes evaluation-only hand mismatch
examples. These diagnostics never enter public model inputs. See the
[replay coverage audit](reports/blb_replay_audit.md) for results and schema limits.

## Expanded replay evaluation

Public-history extraction now has its own adapter, `extract_public_history`.
It resolves only public card identities; opening, draw, and end-of-turn hand
lists contribute lengths only. It stops at unsupported count changes or
tutors. `reconstruct_snapshots` independently audits private identities and
may reject even a count-consistent history. Neither adapter infers missing
public movements from label differences.

Mulligan games are opt-in with `--include-mulligans`. The kept opening hand
must contain exactly `7 - num_mulligans` cards and pass the same identity
audit. Predictions approximate retained hands as uniform samples, ignoring
keep choices, Arena smoothing, and London-mulligan bottoming order. The
default remains the original zero-mulligan experiment.

Expand the sample using the existing compressed prefix, then evaluate:

```bash
.venv/bin/python examples/download_replay_sample.py \
  --prefix data/replay_data_public.BLB.PremierDraft.prefix.gz \
  --games 5000 --output data/replay_sample_5000.BLB.PremierDraft.csv.gz
.venv/bin/python examples/replay_hands.py \
  --replays data/replay_sample_5000.BLB.PremierDraft.csv.gz --max-games 5000 \
  --output reports/blb_hands_5000.json
.venv/bin/python examples/replay_hands.py \
  --replays data/replay_sample_5000.BLB.PremierDraft.csv.gz --max-games 5000 \
  --include-mulligans --output reports/blb_hands_5000_mulligans.json
```

If no local prefix is available, omit `--prefix` to download a bounded range.
Reports now include paired 95% percentile bootstrap intervals for method
differences, resampling whole draft events to retain within-event dependence.
Snapshots still average within games, then games receive equal weight.
Intervals describe uncertainty within the filtered prefix sample; they do not
correct selection bias or establish full-dataset performance. See the
[expanded experiment](reports/blb_hands_5000.md).

## Replay coverage improvements

Unidentified hand departures can now be represented by their **public count**
without giving the model their private identities. For example, if a hand
shrinks from seven to six with no recorded target draw or named departure,
`unknown_departures=1` records the net loss. It does not claim a particular
card was discarded or cast. This is opt-in, so earlier experiments retain their
strict reconstruction behavior.

The independent identity audit still has to pass: after recorded draws and
named departures, the actual hand must be a subset with exactly the specified
number of cards removed. Unexplained additions, replacements, tutors, and
named departures absent from the reconstructed hand still stop evaluation.
Missing gains are never treated as draws just to make counts match.

The hand baseline marginalizes unidentified departures under its existing
random-selection assumption. It retains their unknown identities in a latent
pool and accounts for their count when checking whether future draws are
possible. It does not learn why those cards left the hand. This is a weaker
observation than a named card departure; card-type accounting still does not
establish exact zone provenance.

Compare strict and expanded reconstruction on the same 5,000 replay rows:

```bash
.venv/bin/python examples/audit_replays.py \
  --replays data/replay_sample_5000.BLB.PremierDraft.csv.gz \
  --max-games 5000 --turns 3 10 30 --include-mulligans \
  --compare-unknown-departures --output reports/blb_coverage_improvement.json
```

Run hand inference with the expanded coverage:

```bash
.venv/bin/python examples/replay_hands.py \
  --replays data/replay_sample_5000.BLB.PremierDraft.csv.gz \
  --max-games 5000 --max-turn 30 --include-mulligans \
  --allow-unknown-departures --output reports/blb_hands_expanded_coverage.json
```

The audit distinguishes all available recorded states, accepted hand labels,
and games reaching their recorded ending. Inference can have lower coverage
because no training deck is compatible with a public history; these exclusions
are counted separately. See [the coverage report](reports/blb_coverage_improvement.md).

## Flexible priors and boosted-tree research

The optional research experiment audits finite-prior failures, fits a smoothed
color-group/deck-shape prior, and compares shared boosted card-presence models
with and without public board features. Its event-disjoint training/validation/
test split and two-fold training-prior cross-fitting prevent draft leakage.
Calibration uses validation labels only. Every candidate card is retained;
negative sampling and class reweighting are unnecessary at this scale.

```bash
.venv/bin/python -m pip install -e '.[dev,research]'
.venv/bin/python examples/audit_prior_failures.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python examples/compare_hand_models.py
.venv/bin/python examples/predict_research_hand.py data/public_hand_example.json
```

Use `--train-snapshots`, `--validation-snapshots`, `--test-snapshots`, and
`--iterations` to change the bounded experiment size. The runner saves its local
model artifacts under `data/`; inference takes public-history JSON, not hidden
hands or held-out decklists. Core baselines still need no research dependencies.

Read [the research report](reports/blb_model_comparison.md) for the failure
audit, exact prior assumptions, coverage, fair common-state comparisons,
calibration, results without basic lands, and uncertainty intervals. The
observed Recall@20 rises from 49.8% with the flexible prior to about 53.7–53.8%
with trees on the same 800 test states. Additional board features did not show
a clear gain over history features in this run. These remain research models;
the website's saved examples use the previous baseline.
