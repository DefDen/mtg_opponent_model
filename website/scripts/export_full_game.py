"""Add one fully audited recorded game to the replay explorer.

This is a selected demonstration, not an additional aggregate benchmark.
Every available half-turn must pass the audit and finite deck prior.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from mtg_opponent_model.hands import HandModel, PublicHistory, hand_metrics
from mtg_opponent_model.replays import load_card_names, read_replay_rows, reconstruct_snapshots
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader
try:
    from .board_data import public_board
except ImportError:  # Direct execution from the research repository.
    from board_data import public_board

ROOT = Path(__file__).resolve().parents[2]
GAME_ID = "af90260c3b4d4d63889a008186d85de0:6:1"


def recorded_steps(row: dict[str, str]) -> list[tuple[int, str]]:
    """Enumerate all non-padded public half-turn records independently of labels."""
    sides = ("user", "oppo") if row["on_play"] == "True" else ("oppo", "user")
    available = []
    for turn in range(1, 31):
        for side in sides:
            prefix = f"{side}_turn_{turn}_"
            if any(float(row.get(prefix + f"eot_{p}_life", "0") or 0) != 0 for p in ("user", "oppo")):
                available.append((turn, "target" if side == "user" else "observer"))
    return available


def main() -> None:
    path = ROOT / "website/dist/data.json"
    dataset = json.loads(path.read_text())
    report = dataset["experiments"][1]
    settings = report["settings"]
    rows = read_replay_rows(ROOT / "data/replay_sample_5000.BLB.PremierDraft.csv.gz", settings["max_games"])
    groups = sorted({row["draft_id"] for row in rows})
    random.Random(settings["seed"]).shuffle(groups)
    held_out = set(groups[:max(1, round(0.2 * len(groups)))])
    row = next(row for row in rows if ":".join(row[k] for k in ("draft_id", "match_number", "game_number")) == GAME_ID)
    assert row["draft_id"] in held_out
    if int(row["num_turns"]) >= 30:
        raise ValueError("Game may exceed the replay's 30-round schema")
    names = load_card_names(ROOT / "data/cards.csv")
    snapshots, reason = reconstruct_snapshots(row, names, 30)
    expected = recorded_steps(row)
    actual = [(s.public.steps[-1].turn, s.public.steps[-1].active_player) for s in snapshots]
    if reason or actual != expected or not expected or expected[-1][0] != int(row["num_turns"]):
        raise ValueError(f"Game is not fully reconstructed: {reason}")
    loader = SeventeenLandsLoader(ROOT / "data/game_data_public.BLB.PremierDraft.csv.gz",
                                  "BLB", "PremierDraft", settings["max_deck_rows"])
    train = [deck for deck in loader.load() if deck.group_id not in held_out]
    assert len(train) == report["training_decks"]
    model = HandModel(train)
    universe = sorted({c[5:] for c in row if c.startswith("deck_")} | set(model.cards))
    baseline_by_size = {}
    frames = []
    cumulative = Counter()
    for snapshot in snapshots:
        step = snapshot.public.steps[-1]
        probabilities = model.predict(snapshot.public)  # No private label input.
        cumulative.update(step.departed)
        if step.hand_size not in baseline_by_size:
            baseline_by_size[step.hand_size] = model.predict(PublicHistory(step.hand_size, ()))
        frames.append({"step": asdict(step), "actual_hand": snapshot.actual_hand,
                       "probabilities": {c: p for c, p in probabilities.items() if p > 0},
                       "baseline": {c: p for c, p in baseline_by_size[step.hand_size].items() if p > 0},
                       "compatible_decks": sum(all(deck.cards.get(c, 0) >= n for c, n in cumulative.items()) for deck in train),
                       "metrics": hand_metrics(probabilities, snapshot.actual_hand, universe),
                       "board": public_board(row, names, step.turn, step.active_player)})
    full_game = {"id": "full-game-1", "label": "Full game 1", "on_play": row["on_play"] == "True",
                 "opening_hand_size": snapshots[0].public.opening_hand_size,
                 "frames": frames, "stop_reason": None, "source_game": GAME_ID,
                 "complete": True, "recorded_turns": int(row["num_turns"]),
                 "recorded_snapshot_count": len(expected),
                 "recorded_winner": "target" if row["won"] == "True" else "observer",
                 "end_reason": "not provided by this dataset",
                 "selection": "Selected fully audited held-out game with nonzero prior support through every recorded half-turn"}
    dataset["cases"] = [full_game] + [game for game in dataset["cases"] if game["id"] != full_game["id"]]
    for game in dataset["cases"]:
        if game["id"] != full_game["id"]:
            game["complete"] = False
    path.write_text(json.dumps(dataset, separators=(",", ":")), encoding="utf-8")
    print(f"Exported full game: {len(frames)} recorded snapshots, {row['num_turns']} rounds, winner {full_game['recorded_winner']}")


if __name__ == "__main__":
    main()
