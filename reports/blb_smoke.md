# Bloomburrow historical smoke experiment

Run on 2026-10-11 using the 17Lands Bloomburrow Premier Draft public game
dataset. Source: [17Lands](https://www.17lands.com/public_datasets), CC BY 4.0.
The raw gzip download is 69,271,450 bytes and stays in the ignored `data/`
directory. Its SHA-256 and machine-readable results are in `blb_smoke.json`.

Reproduce:

```bash
.venv/bin/python examples/historical.py \
  data/game_data_public.BLB.PremierDraft.csv.gz \
  --max-rows 20000 --observed-count 2 --seed 42 \
  --output reports/blb_smoke.json
```

The first 20,000 game rows yielded 4,261 distinct deck configurations from
3,445 events after removing 15,739 repeated rows. No incomplete rows occurred.
Deck sizes ranged from 40 to 49 cards. The split kept all configurations of
an event together: 3,441 training decks and 820 test decks, corresponding to
2,756 training events and 689 test events. The training vocabulary has 274
cards. Each test deck contributed one case with two uniformly sampled distinct
observations; the other distinct cards were targets. Copy counts were retained
in loading but the model and metrics use presence, not multiplicity.

| Metric | Co-occurrence | Popularity |
| --- | ---: | ---: |
| Recall@5 | 0.1529 | 0.1058 |
| Recall@10 | 0.2486 | 0.1658 |
| Recall@20 | 0.3924 | 0.2604 |
| MRR | 0.9638 | 0.7236 |
| Target vocabulary coverage | 0.99994 | 0.99994 |

Co-occurrence improved every measured ranking metric on this sample.
Both methods used the same split and cases and excluded already observed
cards. Unknown targets remained in the recall denominators.

This is a pipeline smoke experiment, not a final benchmark. A file prefix may
be biased and can cut off an event. Basic lands are included as observations,
predictions, and targets: frequent lands make a first relevant prediction much
easier, so high MRR does not imply accurate recovery of the whole deck.
Uniform observations do not model reveal order or game actions. Records are
player decks from 17Lands contributors, not complete decks of their opponents.
No uncertainty estimates or tuning were performed. A next experiment should
use the full file and report results with and without basic lands.
