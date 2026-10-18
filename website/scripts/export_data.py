"""Export real held-out replay predictions for the static research explorer.

Run from the research repository: .venv/bin/python website/scripts/export_data.py
This keeps labels separate from the PublicHistory passed to HandModel.
"""
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random

from mtg_opponent_model.hands import HandModel, IncompatibleHistoryError, PublicHistory, hand_metrics
from mtg_opponent_model.replays import load_card_names, read_replay_rows, reconstruct_snapshots
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader
from board_data import public_board

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    report_path = ROOT / "reports/blb_hands_5000.json"
    report = json.loads(report_path.read_text())
    settings = report["settings"]
    rows = read_replay_rows(ROOT / "data/replay_sample_5000.BLB.PremierDraft.csv.gz", settings["max_games"])
    groups = sorted({row["draft_id"] for row in rows})
    random.Random(settings["seed"]).shuffle(groups)
    held_out = set(groups[:max(1, round(0.2 * len(groups)))])
    loader = SeventeenLandsLoader(ROOT / "data/game_data_public.BLB.PremierDraft.csv.gz",
                                  "BLB", "PremierDraft", settings["max_deck_rows"])
    train = [deck for deck in loader.load() if deck.group_id not in held_out]
    assert len(train) == report["training_decks"]
    assert not {deck.group_id for deck in train} & held_out
    names = load_card_names(ROOT / "data/cards.csv")
    cards = {}
    with (ROOT / "data/cards.csv").open() as stream:
        for row in csv.DictReader(stream):
            cards[row["name"]] = {"colors": row["color_identity"], "types": row["types"],
                                  "mana_value": row["mana_value"]}
    model = HandModel(train)
    universe = sorted({column[5:] for column in rows[0] if column.startswith("deck_")} | set(model.cards))
    baseline_by_size = {}
    cases = []
    seen = set()
    for row in rows:
        if row["draft_id"] not in held_out:
            continue
        key = tuple(row[k] for k in ("draft_id", "match_number", "game_number"))
        if key in seen:
            continue
        seen.add(key)
        snapshots, reason = reconstruct_snapshots(row, names, settings["max_turn"])
        if len(snapshots) < 4:
            continue
        frames = []
        cumulative = Counter()
        for snapshot in snapshots:
            try:
                probabilities = model.predict(snapshot.public)
            except IncompatibleHistoryError:
                break
            step = snapshot.public.steps[-1]
            cumulative.update(step.departed)
            size = step.hand_size
            if size not in baseline_by_size:
                baseline_by_size[size] = model.predict(PublicHistory(size, ()))
            baseline = baseline_by_size[size]
            compatible = sum(all(deck.cards.get(card, 0) >= count for card, count in cumulative.items())
                             for deck in train)
            frames.append({"step": asdict(step), "actual_hand": snapshot.actual_hand,
                           "board": public_board(row, names, step.turn, step.active_player),
                           "probabilities": {c: p for c, p in probabilities.items() if p > 0},
                           "baseline": {c: p for c, p in baseline.items() if p > 0},
                           "compatible_decks": compatible,
                           "metrics": hand_metrics(probabilities, snapshot.actual_hand, universe)})
        if len(frames) < 4:
            continue
        cases.append({"id": f"replay-{len(cases) + 1}", "label": f"Replay {len(cases) + 1}",
                      "on_play": row["on_play"] == "True", "opening_hand_size": 7,
                      "frames": frames, "stop_reason": reason,
                      "source_game": ":".join(key)})
        if len(cases) == 8:
            break
    payload = {"experiments": [json.loads((ROOT / "reports" / name).read_text()) for name in
                               ("blb_hands_smoke.json", "blb_hands_5000.json", "blb_hands_5000_mulligans.json")],
               "cases": cases, "cards": {c: cards.get(c, {}) for c in universe},
               "explorer_experiment": "blb_hands_5000", "training_decks": len(train),
               "selection": "First eight held-out games with at least four supported early half-turn snapshots",
               "source": "17Lands public datasets • CC BY 4.0"}
    payload["board_note"] = "Public end-of-half-turn battlefield snapshots. Tapping, counters, attachments, and exact action order are unavailable."
    dest = ROOT / "website/dist/data.json"
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(cases)} real games / {sum(len(c['frames']) for c in cases)} snapshots to {dest}")


if __name__ == "__main__":
    main()
