# Flexible deck priors and shared boosted hand prediction

This experiment addresses all four follow-ups: diagnose finite-prior failures,
replace exact-deck support with a generative prior, measure availability and
probability quality together, and test boosted trees with public history and
board features. Source: [17Lands public datasets](https://www.17lands.com/public_datasets),
CC BY 4.0. No neural network or website model replacement is involved.

## What caused the prior failures?

The [failure audit](blb_prior_failure_audit.json) reproduces the earlier
12,448 audited held-out states with the original 4,180-deck prior:

| Outcome | States |
| --- | ---: |
| Compatible historical deck | 8,644 |
| Individually supported cards/copy counts, but unseen combination | 3,790 |
| Exceeds the recorded deck's single-deck accounting | 14 |

Thus 3,790 of 3,804 failures are failures of finite deck-list support, rather
than an individually unseen card or an unusually large copy count. The final
14 are flagged for possible card reuse, generated cards, or accounting issues;
the audit cannot establish their cause. Four already-supported states also
exceed their recorded deck's accounting, for **18 flagged states in total**.
These diagnostics use recorded decks only for evaluation, never inference.
A consistent recorded-deck check is not proof of complete zone semantics.

## Flexible prior

The new prior is a color-group mixture over training-derived deck shapes.
A shape fixes deck size and land/spell counts. Within each category, a smoothed
Dirichlet-multinomial distribution generates card copy counts. This supports
new card combinations while preserving finite deck size and category slots.
Limited has no general four-copy restriction; copies cannot exceed the finite
slots. All named departures consume those slots without replacement.

Inference analytically integrates possible deck counts and current hands; it
does not sample or repair test decks. Public revealed counts update color
weights and card distributions. The known legal card pool receives a small
positive floor for previously unseen legal cards. Fixed settings are
concentration 50, global shrinkage 50, and copy-frequency floor 0.01.
Color pairs are not learned tribal/archetype labels. Draws and departures
retain the earlier uniform/random assumptions.

For the new train/validation/test split, both validation and test events are
excluded from prior fitting, leaving 4,096 prior decks. On all 12,448 audited
test states:

| Coverage | Historical prior | Flexible prior |
| --- | ---: | ---: |
| Prediction available | 8,631 (69.3%) | 12,448 (100%) |
| Among 12,430 recorded-deck-consistent states | 8,627 (69.4%) | 12,430 (100%) |

The 18 flagged accounting states remain flagged; availability does not mean
that their movement mechanics have been solved. The historical count differs
from 8,644 because the new experiment additionally excludes validation events.
Hand reconstruction checks were not loosened by this prior change.

## Evaluation design

The source is the same first 5,000 replay games, covering 1,025 draft events.
Seed 42 assigns 615 events to training, 205 to validation, and 205 to test.
No event crosses a boundary, including repeated games or deck revisions.

For a bounded first tree experiment, one seeded random audited state is
selected per game, then seeded game samples select **1,500 training**, **400
validation**, and **800 test** cases. All 276 candidate cards are retained:
414,000 training card rows, with no negative sampling or class weighting.
Accuracy numbers describe these 800 states, while coverage numbers above
describe every audited test state. Mulligan kept-hand approximation and
unidentified-departure marginalization are enabled.

Two event folds cross-fit the training prior features: an event and all other
events in its fold are excluded from the deck prior used to generate its
supervised training features. Validation and test events are excluded from
all priors. Hyperparameters are fixed, and early stopping is disabled to avoid
row-level splits of correlated games. Validation labels alone fit the optional
Platt sigmoid calibration. Test labels are not used for fitting or calibration.

The shared histogram-boosting classifier scores `(public history, candidate
card)` pairs. There is no separate classifier or ID feature per card. Its
24 history/card features include the flexible prior probability, revealed
copies/recency, turn/hand/draw counts, colors, mana value, and card types.
The board variant adds 15 public features: life totals, permanent counts,
candidate presence on either board, and target land-color counts. It receives
no hidden draw identities, actual hand, recorded deck, or future board state.
Candidate scores do not define a coherent joint hand distribution. Hard zero
probabilities from finite slots/empty hands are preserved, but the sum of tree
marginals is not constrained to hand size.

## Results on the same 800 test cases

Brier and binary log loss are averaged over the fixed card universe. Lower is
better. ECE uses ten equal-width probability bins and includes every negative
card/case pair. Empty target hands receive zero recall/MRR. Basic lands are
included in this first table.

| Model | Recall@5 | Recall@10 | Recall@20 | Brier ↓ | Log loss ↓ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-size-only flexible prior | 21.3% | 25.0% | 34.9% | 0.01342 | 0.06516 | 0.00323 |
| Flexible movement prior | 25.1% | 34.7% | 49.8% | 0.01280 | 0.05604 | 0.00271 |
| Boosted history, raw | 28.0% | 38.3% | 53.8% | 0.01248 | 0.05447 | 0.00042 |
| Boosted history, calibrated | 28.0% | 38.3% | 53.8% | 0.01249 | 0.05446 | 0.00047 |
| Boosted board, raw | 28.4% | 38.7% | 53.7% | 0.01249 | 0.05448 | 0.00061 |
| Boosted board, calibrated | 28.4% | 38.7% | 53.7% | 0.01250 | 0.05447 | 0.00056 |

An all-absent predictor has Brier 0.01420 and zero recall, which helps put these
small probability errors in context. Sigmoid calibration gives mixed changes;
it is not automatically an improvement. It is monotonic here, so it does not
change rankings.

The calibrated board model's Recall@20 improvement over the flexible prior is
**3.91 percentage points**, with a 95% paired draft-event bootstrap interval
of **2.43–5.49 points** (1,000 resamples). Its Brier difference is -0.000300
(interval -0.000374 to -0.000229). These intervals describe sampling variation
within this prefix dataset, not cross-set generalization.

Board features do **not** provide a clear incremental gain over the calibrated
history model: Recall@20 difference -0.05 points, interval **-0.81 to +0.77**.
The board Brier difference also includes zero. These board features are coarse
counts, not a representation of card rules, legal plays, or decision windows.

## Comparable historical-prior subset and rescued states

The historical prior can predict only 617 of the 800 sampled states. Comparing
its own 617-case average directly with another model's 800-case average would
confound model quality with coverage. On their shared 617 cases:

| Model | Recall@20 | Brier ↓ |
| --- | ---: | ---: |
| Historical prior | 50.4% | 0.014445 |
| Flexible prior | 51.7% | 0.014345 |
| Boosted history, raw | 54.8% | 0.014076 |
| Boosted board, calibrated | 54.6% | 0.014084 |

The flexible-prior Recall@20 difference on this subset is +1.37 points, with
an interval of -0.12 to +2.81: coverage improves clearly, but ranking improvement
from smoothing alone is uncertain. Its Brier improvement does exclude zero.

On the **183 rescued states**, Recall@20 is 43.2% for the flexible prior,
50.2% for raw history trees, and 50.6% for calibrated board trees. These are
not zero-filled historical predictions; the historical prior abstains there.
None of the 800 sampled cases happens to be among the 18 flagged accounting
states, but the full-state coverage report keeps those flags explicitly.

Removing all five basic land types from candidates and targets gives
Recall@20 of **40.3%** for the flexible prior, **43.7%** for raw history trees,
and **44.2%** for calibrated board trees. The observed gain is not solely due
to guessing basic lands.

## Reproduce and inspect

```bash
.venv/bin/python -m pip install -e '.[dev,research]'
.venv/bin/python examples/audit_prior_failures.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python examples/compare_hand_models.py
.venv/bin/python examples/predict_research_hand.py data/public_hand_example.json
.venv/bin/python -m pytest -q
```

The comparison writes `data/hand_research_models.joblib` (priors, both tree
models/calibrators, card metadata, and feature names), a public-only inference
example, and event partitions. Its JSON report records checksums, package
versions, hyperparameters, calibration bins, bootstrap differences, and real
rescued examples. Artifact reloads reproduce saved predictions. Joblib
artifacts are local Python objects and should only be loaded from a trusted
training run in a compatible environment.

See [blb_model_comparison.json](blb_model_comparison.json) for exact results and
[scikit-learn's classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
and [calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
for the implementation used. The website continues to display its existing
saved baseline predictions; this research run does not silently promote a new
model or rewrite earlier benchmark reports.

The strongest next candidate from this bounded experiment is the simpler
history-only boosted model over the flexible prior. Before production use,
repeat across sets and time periods, test card-effect representations, and
expand movement semantics. Passing and strategic play likelihoods are still
not explicitly modeled.
