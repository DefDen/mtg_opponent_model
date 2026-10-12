# Replay reconstruction coverage audit

Run on 2026-10-12 using the existing 500-game Bloomburrow Premier Draft
prefix sample. Data attribution: 17Lands, CC BY 4.0. This audit includes all
sampled games, unlike the held-out prediction experiment. It measures label
reconstruction coverage, not prediction performance.

Reproduce with:

```bash
.venv/bin/python examples/audit_replays.py
```

Detailed stop locations, mismatch sizes, and evaluation-only examples are in
[blb_replay_audit.json](blb_replay_audit.json).

| Maximum round | Valid snapshots | Games with snapshots | Games without a reconstruction stop |
| --- | ---: | ---: | ---: |
| 3 | 2,479 | 434 | 373 |
| 10 | 4,801 | 434 | 162 |
| 30 | 4,891 | 434 | 160 |

Games without a stop include games that ended before the requested round.
Earlier valid snapshots are retained when a later half-turn fails accounting.
Sixty-five games are excluded for mulligans at every horizon.

Through round three, 53 games stop at unaccounted hand movements, six at
tutors, and three at departures absent from the reconstructed hand. Of the
53 mismatches, 42 occur on the observer's turn. Thirty-one mismatches add one
card with no missing cards; eighteen remove one card with no added cards.
These differences describe labels and do not establish the underlying action.

The schema has `cards_drawn` and `cards_tutored` on target turns, but only
`cards_drawn_or_tutored` on observer turns. It lacks separate target draw,
discard, and flash-creature columns on observer turns. An end-of-turn hand
difference cannot establish a public draw count or casting zone provenance.
The audit therefore records failures without turning label differences into
model inputs or relaxing the exact hand check.

Extending the horizon alone roughly doubles available snapshots, but many
more games stop. Coverage after round ten adds only 90 snapshots in this
sample. Reliable later-turn evaluation needs richer public event provenance
or explicitly justified movement handling. These prefix-sample results do
not establish coverage for the full dataset.
