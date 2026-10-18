"""Add public battlefield snapshots without recalculating saved predictions."""
import json
from pathlib import Path

from mtg_opponent_model.replays import load_card_names, read_replay_rows
from board_data import public_board

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = ROOT / "website/dist/data.json"
    dataset = json.loads(path.read_text())
    names = load_card_names(ROOT / "data/cards.csv")
    rows = read_replay_rows(ROOT / "data/replay_sample_5000.BLB.PremierDraft.csv.gz", 5000)
    by_game = {":".join(row[k] for k in ("draft_id", "match_number", "game_number")): row for row in rows}
    visible = set()
    for game in dataset["cases"]:
        row = by_game[game["source_game"]]
        for frame in game["frames"]:
            step = frame["step"]
            frame["board"] = public_board(row, names, step["turn"], step["active_player"])
            for player in frame["board"].values():
                for zone in ("lands", "creatures", "other"):
                    visible.update(player[zone])
    dataset["board_note"] = "Public end-of-half-turn battlefield snapshots. Tapping, counters, attachments, and exact action order are unavailable."
    path.write_text(json.dumps(dataset, separators=(",", ":")), encoding="utf-8")
    print(f"Added public boards to {sum(len(g['frames']) for g in dataset['cases'])} frames")
    print("Additional artwork needed:", sorted(visible - json.loads((ROOT / "website/dist/card-images.json").read_text()).keys()))


if __name__ == "__main__":
    main()
