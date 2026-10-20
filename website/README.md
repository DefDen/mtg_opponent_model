# Hidden Hand research explorer

A buildless static website showing saved held-out replay predictions, with
on-demand browser inference for uploaded supported replays. The export contains one full recorded game
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
between snapshots; they do not reconstruct exact action order. The historical baseline uses board state as context; the board-tree model
also conditions on public board features.

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

The replay sidebar shows the three largest probability rises and falls across the
full candidate pool between successive opponent turn-end states. Click a change or prediction
to inspect its probability history through the selected state (future values stay
hidden). Differences use only exported model probabilities, never hand labels.
The first opponent state has no preceding opponent comparison. Deltas and the change panel remain fixed during observer turns, even if current probabilities change. These are combined updates
from public hand movements, not causal attribution to individual actions or a
learned model of strategic choices. The default remains the interpretable historical-deck baseline. A method selector
and comparison panel also expose the hand-size baseline, smoothed prior, and raw
boosted-history and boosted-board models on the same states.

Hover a battlefield card, prediction, probability-change row, revealed hand card,
or named departure to see a large card preview. Keyboard focus also opens the
preview; Escape dismisses it. Previews stay inside the viewport and clear when
the replay changes or the page scrolls. Hidden card backs never reveal identities.
Clicking card images still opens the full-size dialog on desktop and touch devices.

To refresh the comparison, run `.venv/bin/python website/scripts/export_methods.py`
after exporting replay states, then run the card-art exporter. The comparison
exporter checks the artifact checksum against the research report and verifies
that every displayed draft belongs to the saved research test split. Inference
accepts only public history and public board fields. Trained artifacts stay local;
the site receives precomputed probabilities. Different training pools and model
assumptions make this a demonstration, not a replacement for the paired benchmark.

## Importing a new replay

On the hosted site, expand **Import a replay**, upload a plain wide 17Lands replay
CSV or public-actions JSON, choose a CSV game when needed, then click **Run
predictions**. This scores a new replay in a module Web Worker using the existing
fitted models. It requires no inference server or republishing. Inputs remain in
browser memory and are not uploaded or saved. Reloading clears the imported game.
The roughly 3.2 MB model table loads only when inference is requested. Cancellation
terminates the worker. Imports do not train or update model parameters.

The trained pool is **BLB**, with Special Guests present in the original dataset.
Other sets, Arena logs, 17Lands replay URLs/codes, compressed CSV uploads, and
long-form (one turn per row) CSV are not supported. CSV is the existing wide
`user_turn_N_...` schema, one game per row, with `expansion=BLB`. Limit: 8 MB and
1,000 rows per file; select one game. JSON uses schema version 1; download
`public-replay-example.json` for a valid example containing only public fields.
Up to 60 consecutive half-turn states through turn 30 are accepted.

The CSV `user` is the modeled target (labeled Opponent on the board), which may be
the uploader's own recorded perspective. Private opening, draw, and hand lists
supply only their **lengths**; identities and recorded decks never enter inference.
JSON requires explicit hand sizes and anonymous draw counts. Board life, land,
creature, and other-permanent lists must be supplied; absent zones are empty in
the CSV schema. Hand labels are unavailable for imports, so Reveal is disabled.

CSV imports stop before tutors and inconsistent hand movements. They do not infer
unidentified departures. JSON may provide explicit unknown-departure counts.
These are count-consistent public histories, not independently identity-audited
games; source zones, returns, generated cards, and same-size substitutions may
remain unresolved. A processed prefix is never labeled a complete game. Unknown
out-of-pool departures are rejected. An unsupported historical prior is shown as
Unavailable, while smoothed and tree methods can continue. Imported games may
belong to training events and are never included in benchmark reports.

Browser inference exports the original historical deck prior, smoothed mixture
parameters, card metadata/ID mapping, and fitted raw tree nodes. Refresh with:

```bash
.venv/bin/python website/scripts/export_browser_models.py
.venv/bin/python website/scripts/fetch_card_art.py --all-pool
node --test website/tests/*.test.js
```

Tests compare all five JavaScript methods against the saved Python predictions
on every displayed state within 1e-9. Tree export uses scikit-learn internals, so
regenerate and rerun parity checks after changing the trained artifact or library
version. Card art for the trained pool and BLB tokens is bundled ahead of time;
unrecognized board tokens fall back to names. New inference tables contain
historical training decks, not the uploaded target's decklist.
