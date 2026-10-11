# Hand prediction from public replay movements

Run on 2026-10-11. Source: [17Lands public datasets](https://www.17lands.com/public_datasets),
CC BY 4.0. This experiment predicts current hand presence rather than deck
membership. The numerical results and three inspectable examples are in
[blb_hands_smoke.json](blb_hands_smoke.json).

Reproduce after downloading the inputs described in the README:

```bash
.venv/bin/python examples/replay_hands.py --output reports/blb_hands_smoke.json
```

The input is the first 500 complete Bloomburrow Premier Draft replay game rows,
extracted from the first 8 MiB of the public gzip. The actual format is a wide
row per game containing half-turn summaries, not a fully ordered action log.
The card mapping converts printing-specific Arena IDs to names. Checksums
identify the files used; gzip container timestamps can differ on redownload.

Of 110 replay events, 22 were held out (seed 42), containing 104 games. The
prior uses 4,230 deck configurations from the first 20,000 historical game
rows, with every held-out event excluded. Training/test event overlap is zero.
We evaluate through the first three rounds, ending each snapshot after a
half-turn. Zero-mulligan games only; basic lands included.

There are 520 evaluated snapshots across 92 games. Twelve games were excluded
for mulligans. Twelve games stopped at unaccounted hand movements, and one
stopped at a departure absent from the reconstructed hand; earlier valid
snapshots were retained. Two snapshots had no compatible training deck and
were excluded from both comparisons. These categories need not partition the
92 evaluated games because stopping can occur after valid snapshots.

| Metric | Public movements | Hand size only |
| --- | ---: | ---: |
| Recall@5 | 0.2969 | 0.2634 |
| Recall@10 | 0.3764 | 0.3025 |
| Recall@20 | 0.5120 | 0.4035 |
| MRR | 0.7446 | 0.5070 |
| Brier score (lower is better) | 0.01610 | 0.01734 |

All metrics average snapshots within each game, then give each game equal
weight. Brier averages binary presence errors over 276 schema-defined card
types. Most types are absent, so a small Brier score alone is not evidence of
good calibration. Empty hands contribute zero recall/MRR and still contribute
probability error. Only positive-probability cards enter rankings.

## Mechanism and private-information boundary

Each training deck configuration has equal prior weight. For each public card
leaving the target's hand, the model updates deck weights using the remaining
copy count divided by remaining deck size, then removes that copy. Within a
compatible deck, the residual hand is assumed uniformly distributed among
remaining cards. The hypergeometric probability of holding at least one copy
is averaged over the posterior deck weights.

The model sees only `PublicHistory`: initial hand size, anonymous draw counts,
public departure identities, and current hand size. It never receives the
held-out deck, opening-hand identities, private draw identities, or target
labels. Actual end-of-turn hands are reconstructed and checked in the loader
only. Counts come from lengths of recorded lists; their identities remain
private. The tracked player is treated as the prediction target from their
opponent's perspective, since their own hand supplies evaluation labels.

In the first example, the target draws one unknown card and plays Mountain,
leaving seven cards. Mountain is still predicted at 74.1% because additional
copies may be held. After Forest and Brazen Collector are played on turn two,
Rust-Shield Rampager enters the top five at 11.9%; it is actually in hand.
The JSON shows all three early target-turn examples, including wrong guesses.

These are movement-conditioned predictions under a random-departure policy,
not a model of strategic decisions. Library effects, Arena opening-hand
smoothing, and casting zone provenance are not modeled. Missing movements
cause truncation; matching counts do not guarantee complete zone semantics.
The prefix sample, early turns, and filtering limit generalization. The next
step is more reliable movement coverage before fitting action likelihoods.
