"""Measure replay reconstruction coverage before expanding hand evaluation."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path

from mtg_opponent_model.replays import load_card_names, read_replay_rows, reconstruct_snapshots


def audit(rows, names, max_turn, allow_mulligans=False):
    stops = Counter()
    stop_locations = Counter()
    mismatches = Counter()
    snapshots_count = games_with_snapshots = 0
    examples = []
    seen = set()
    for row in rows:
        game_id = ":".join(row[k] for k in ("draft_id", "match_number", "game_number"))
        if game_id in seen:
            continue
        seen.add(game_id)
        details = {}
        snapshots, reason = reconstruct_snapshots(row, names, max_turn, diagnostics=details,
                                                 allow_mulligans=allow_mulligans)
        snapshots_count += len(snapshots)
        games_with_snapshots += bool(snapshots)
        if reason is None:
            continue
        stops[reason] += 1
        if "turn" in details:
            stop_locations[f"{details['turn']}:{details['active_player']}:{reason}"] += 1
        if reason == "unaccounted_hand_movement":
            added = sum(details["added_cards_evaluation_only"].values())
            missing = sum(details["missing_cards_evaluation_only"].values())
            mismatches[f"added={added},missing={missing}"] += 1
        if len(examples) < 10 and reason != "mulligan":
            examples.append({"game_id": game_id, "reason": reason,
                             "valid_snapshots_before_stop": len(snapshots), **details})
    return {"max_turn": max_turn, "allow_mulligans": allow_mulligans,
            "unique_games": len(seen),
            "games_with_snapshots": games_with_snapshots,
            "valid_snapshots": snapshots_count,
            "games_without_reconstruction_stop": len(seen) - sum(stops.values()),
            "stops_by_reason": dict(sorted(stops.items())),
            "stops_by_turn_player_reason": dict(sorted(stop_locations.items())),
            "unaccounted_movement_sizes": dict(sorted(mismatches.items())),
            "stop_examples_evaluation_only": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=Path("data/replay_sample.BLB.PremierDraft.csv.gz"))
    parser.add_argument("--cards", type=Path, default=Path("data/cards.csv"))
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--turns", type=int, nargs="+", default=[3, 10, 30])
    parser.add_argument("--include-mulligans", action="store_true",
                        help="Report both zero-mulligan and mulligan-inclusive coverage")
    parser.add_argument("--output", type=Path, default=Path("reports/blb_replay_audit.json"))
    args = parser.parse_args()
    if any(turn < 1 or turn > 30 for turn in args.turns):
        parser.error("turns must be between 1 and 30")
    rows = read_replay_rows(args.replays, args.max_games)
    names = load_card_names(args.cards)
    def checksum(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {"source": "17Lands public datasets, CC BY 4.0",
              "inputs": {"replays": str(args.replays), "cards": str(args.cards)},
              "input_sha256": {str(path): checksum(path) for path in (args.replays, args.cards)},
              "replay_rows": len(rows),
              "scope": "All sampled games; reconstruction coverage, not prediction performance",
              "limitations": ["File-prefix sample", "Private labels used only for evaluation diagnostics",
                              "Matching hands do not establish casting zone provenance",
                              "No missing movements are inferred from private labels"],
              "audits": [audit(rows, names, turn, allow_mulligans)
                         for turn in dict.fromkeys(args.turns)
                         for allow_mulligans in ([False, True] if args.include_mulligans else [False])]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for result in report["audits"]:
        print(json.dumps({key: result[key] for key in (
            "max_turn", "allow_mulligans", "unique_games", "games_with_snapshots",
            "valid_snapshots", "games_without_reconstruction_stop", "stops_by_reason")}))
    print(f"Evaluation-only stop examples: {args.output}")


if __name__ == "__main__":
    main()
