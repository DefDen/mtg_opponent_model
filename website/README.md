# Hidden Hand research explorer

A buildless static website showing real held-out replay predictions. No live
inference runs in the browser. The export contains one full recorded game
(17 half-turn states through turn 9) and eight early-game excerpts (48 states)
from the 5,000-row no-mulligan experiment, plus the aggregate reports for all
three existing hand experiments.

The battlefield view places both players' public lands, creatures, other
permanents, and life totals on a playmat, with hand predictions alongside it.
Choose a game, step through half-turns, or use play/pause. Newly appearing
cards animate onto the board; duplicate copies remain separate and cards
absent from a later snapshot leave the board. Click any card for a full preview.
Reveal the opponent's actual hand to inspect successes and misses. The
observer's hand remains unknown because its identities are not in this data.
Research progress retains the experiment comparisons and model limitations.

Board state comes from public end-of-half-turn replay fields, not cumulative
played-card guesses or private labels. Tapping, counters, attachments, and
within-turn event order are unavailable. Animations illustrate transitions
between snapshots; they do not reconstruct exact action order. Battlefield
context is displayed only: the model's saved predictions are unchanged and do
not condition on these board fields.

From the research repository, regenerate saved predictions:

```bash
.venv/bin/python website/scripts/export_data.py
.venv/bin/python website/scripts/export_full_game.py
```

This uses local research data and the existing hand model, excludes all
held-out draft IDs from training, and calls the model with public inputs only.
Labels are exported separately for inspection. It requires the 5,000-row
replay sample and reports already present in the research checkout.
To add public board context to an existing prediction export without running
inference again, use `.venv/bin/python website/scripts/enrich_board.py`.

The complete game is the first replay option. Its export independently checks
that every available non-padded half-turn is included, the final turn matches
the game summary, every hand label passes reconstruction, and every prediction
has compatible prior decks. Its draft is excluded from training. This is a
selected supported demonstration, not a new unbiased benchmark. The recorded
result is displayed only at the end. The dataset does not identify the ending
action; the last board state can still show both players with positive life.

Check website code from this directory:

```bash
node --check dist/app.js
node --test tests/*.test.js
```

The persistent hosted address is:
https://mtg-hand-prediction-lab.sharpwave9.chatgpt.site
It is a private deployed site, independent of a local preview server.

For local viewing from this directory:

```bash
npm start
```

Then open http://localhost:8000. Opening index.html as a file will not allow
the browser to fetch data.json reliably.

The Sites manifest declares private hosting. Local browser automation was
unavailable in this session; validation covers JavaScript syntax, real exported
snapshot consistency, ranking calculations, and the Python test suite.

Data: 17Lands public datasets, CC BY 4.0. Full card images are downloaded from
Scryfall and bundled in `dist/assets/cards/`, with attribution and source links
in `card-images.json`. No API call or image-provider connection is needed
while viewing a replay. Hidden hands retain original placeholder backs.
Images are illustrative BLB printings, not guaranteed exact Arena variants.
Magic: The Gathering card art is copyright Wizards of the Coast.

After changing the exported replay sample, refresh artwork from the research
repository:

```bash
.venv/bin/python website/scripts/fetch_card_art.py
```
