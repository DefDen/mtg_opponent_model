# Expanded hand prediction and mulligan coverage

Run on 2026-10-12. Data attribution: [17Lands public datasets](https://www.17lands.com/public_datasets),
CC BY 4.0. Inputs are the first 5,000 complete Bloomburrow Premier Draft replay
rows extracted from the existing 8 MiB compressed prefix, the card mapping,
and the first 20,000 historical game rows. Input checksums, settings, and
evaluation-only examples are in the JSON reports.

## Public extraction and supported openings

`extract_public_history` now builds count-consistent public inputs separately
from label auditing. Only public departure identities are resolved. Private
opening, draw, and end-of-turn hand IDs contribute lengths only. Unknown
private IDs do not prevent public extraction. A count-consistent history can
still fail the independent exact identity audit in `reconstruct_snapshots`.
Neither path supplies missing movements from private hand differences.

Mulligans can be enabled explicitly. The recorded kept opening must contain
exactly `7 - num_mulligans` cards, and every retained snapshot passes the same
exact identity check. The model uses a uniform retained-hand approximation;
keep decisions, Arena smoothing, and London-mulligan bottoming order are
unmodeled. Default behavior continues to exclude mulligans.

## Held-out prediction results

Both runs use seed 42 and the same split: 1,025 replay draft events, 205 held
out, and 971 held-out games. All held-out draft IDs are excluded from the
4,180 training deck configurations; training/test event overlap is zero.
Predictions cover the first three rounds and include basic lands.

| Cohort | Evaluated games | Evaluated snapshots | Recall@10, public movements | Recall@10, hand size only | Paired gain, 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero mulligans | 822 | 4,656 | 38.01% | 29.14% | 8.87 points [7.57, 10.07] |
| Mulligans included | 967 | 5,472 | 38.56% | 29.36% | 9.20 points [7.96, 10.41] |

Opt-in reconstruction adds 137 one-mulligan games and eight two-mulligan
games: 145 games and 816 snapshots, about 17.5% more snapshots. These cohort
differences describe expanded coverage, rather than a change to the scoring
model. Paired method comparisons within each cohort use identical snapshots.

For the mulligan-inclusive run:

| Metric | Public movements | Hand size only | Paired difference, 95% interval |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.28648 | 0.25094 | 0.03554 [0.02859, 0.04256] |
| Recall@10 | 0.38556 | 0.29361 | 0.09196 [0.07964, 0.10407] |
| Recall@20 | 0.51625 | 0.38820 | 0.12805 [0.11277, 0.14454] |
| MRR | 0.69600 | 0.51895 | 0.17705 [0.14884, 0.20280] |
| Brier, lower is better | 0.01567 | 0.01666 | -0.000989 [-0.001099, -0.000873] |

Metrics average snapshots within games, then give each game equal weight.
The 95% percentile intervals use 1,000 paired bootstrap resamples of whole
draft events, seed 42. All games from a sampled draft travel together;
method pairs use the same resample. There are 203 evaluated events in the
zero-mulligan run and 205 in the inclusive run.

The inclusive run has 99 hand-movement mismatch stops, three departures
absent from hand, and seven tutor stops. Earlier valid snapshots are retained.
Seventy no-compatible-prior snapshots are excluded from both methods.
Stops can occur in evaluated games and do not partition the game count.

JSON results: [zero mulligans](blb_hands_5000.json),
[mulligans included](blb_hands_5000_mulligans.json).

## Reconstruction coverage across the full sample

The separate [audit](blb_replay_audit_5000.json) covers all 5,000 sampled games,
including training events. Counts below are audited labels before checking
compatibility with the training-deck prior.

| Maximum round | Zero-mulligan snapshots | Mulligan-inclusive snapshots | Inclusive games without a reconstruction stop |
| --- | ---: | ---: | ---: |
| 3 | 24,501 | 28,548 | 4,396 |
| 10 | 48,704 | 56,827 | 2,099 |
| 30 | 49,547 | 57,810 | 2,037 |

Games without a stop include games ending before the requested round. In
the inclusive audit, 4,973 games yield at least one snapshot. One nominally
zero-mulligan game has a six-card recorded opening and is rejected.

Through round three, 540 games fail hand accounting. Of these, 413 stop on
the observer's turn. The `oppo_turn_N_cards_drawn_or_tutored` field describes
the observer's count and cannot supply target draws. Separate target draw,
discard, and flash-creature movements are absent from those half-turns.
Hand differences identify gaps but do not establish the underlying public
action or source zone. These gaps continue to stop reconstruction.

## Reproduction and limits

Commands are in the [README](../README.md#expanded-replay-evaluation). Audit:

```bash
.venv/bin/python examples/audit_replays.py \
  --replays data/replay_sample_5000.BLB.PremierDraft.csv.gz --max-games 5000 \
  --include-mulligans --output reports/blb_replay_audit_5000.json
```

The original 500-game experiment was rerun after the extraction refactor.
Metrics match within floating-point tolerance; evaluated counts, stop reasons,
unsupported-prior counts, and inspectable examples match the saved smoke run.

The larger experiment supports the public-movement baseline within this
filtered early-turn prefix sample. Bootstrap intervals do not remove prefix
or reconstruction selection bias. Matching hands cannot prove that casts
came from hand. Library effects, strategic departure selection, keep choices,
and passes remain unmodeled. Later-turn coverage needs richer public event
provenance before it can support stronger claims about gameplay inference.
