"""Evaluate public-history hand probabilities on audited replay snapshots."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from mtg_opponent_model.hands import HandModel, IncompatibleHistoryError, PublicHistory, hand_metrics
from mtg_opponent_model.hand_evaluation import summarize_hand_scores
from mtg_opponent_model.replays import load_card_names, read_replay_rows, reconstruct_snapshots
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=Path("data/replay_sample.BLB.PremierDraft.csv.gz"))
    parser.add_argument("--cards", type=Path, default=Path("data/cards.csv"))
    parser.add_argument("--decks", type=Path, default=Path("data/game_data_public.BLB.PremierDraft.csv.gz"))
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--max-deck-rows", type=int, default=20000)
    parser.add_argument("--max-turn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--include-mulligans", action="store_true",
                        help="Audit smaller kept hands; use a uniform retained-hand approximation")
    parser.add_argument("--allow-unknown-departures", action="store_true",
                        help="Marginalize unidentified losses derived from public hand-size counts")
    parser.add_argument("--output", type=Path, default=Path("data/replay_hand_results.json"))
    args = parser.parse_args()
    if args.max_turn < 1 or args.max_turn > 30:
        parser.error("max-turn must be between 1 and 30")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    names = load_card_names(args.cards)
    rows = read_replay_rows(args.replays, args.max_games)
    if any(row["expansion"] != "BLB" or row["event_type"] != "PremierDraft" for row in rows):
        raise ValueError("This initial experiment supports BLB PremierDraft")
    groups = sorted({row["draft_id"] for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two replay events")
    random.Random(args.seed).shuffle(groups)
    test_groups = set(groups[:max(1, round(0.2 * len(groups)))])
    loader = SeventeenLandsLoader(args.decks, "BLB", "PremierDraft", args.max_deck_rows)
    all_decks = list(loader.load())
    train = [deck for deck in all_decks if deck.group_id not in test_groups]
    model = HandModel(train)
    # Fixed card universe comes from schema, not the evaluated hand contents.
    universe = sorted({column[5:] for column in rows[0] if column.startswith("deck_")}
                      | set(model.cards))
    scores: dict[str, dict[str, list[dict[str, float]]]] = {
        "public_movements": defaultdict(list), "no_history": defaultdict(list)}
    stops: Counter[str] = Counter()
    unsupported = 0
    snapshots_count = 0
    examples = []
    used_games = set()
    seen_games = set()
    game_groups = {}
    baseline_by_size = {}
    evaluated_games_by_mulligans = Counter()
    for row in rows:
        if row["draft_id"] not in test_groups:
            continue
        game_key = tuple(row[k] for k in ("draft_id", "match_number", "game_number"))
        if game_key in seen_games:
            continue
        seen_games.add(game_key)
        snapshots, reason = reconstruct_snapshots(row, names, args.max_turn,
                                                 allow_mulligans=args.include_mulligans,
                                                 allow_unknown_departures=args.allow_unknown_departures)
        if reason:
            stops[reason] += 1
        for snapshot in snapshots:
            try:
                probabilities = model.predict(snapshot.public)
            except IncompatibleHistoryError:
                unsupported += 1
                continue
            hand_size = snapshot.public.steps[-1].hand_size
            if hand_size not in baseline_by_size:
                baseline_by_size[hand_size] = model.predict(PublicHistory(hand_size, ()))
            baseline = baseline_by_size[hand_size]
            scores["public_movements"][snapshot.game_id].append(
                hand_metrics(probabilities, snapshot.actual_hand, universe))
            scores["no_history"][snapshot.game_id].append(
                hand_metrics(baseline, snapshot.actual_hand, universe))
            if snapshot.game_id not in used_games:
                evaluated_games_by_mulligans[row["num_mulligans"]] += 1
            used_games.add(snapshot.game_id)
            game_groups[snapshot.game_id] = snapshot.group_id
            snapshots_count += 1
            if len(examples) < 3 and snapshot.public.steps[-1].active_player == "target":
                examples.append({"game_id": snapshot.game_id,
                                 "public_input": asdict(snapshot.public),
                                 "actual_hand_evaluation_only": snapshot.actual_hand,
                                 "top_predictions": sorted(probabilities.items(),
                                                           key=lambda x: (-x[1], x[0]))[:10]})
    if not snapshots_count:
        raise ValueError("No supported evaluation snapshots")
    summary = summarize_hand_scores(scores, game_groups,
                                    bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    report = {
        "source": "17Lands public datasets, CC BY 4.0",
        "replay_source_url": "https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.BLB.PremierDraft.csv.gz",
        "input_sha256": {str(path): checksum(path) for path in (args.replays, args.cards, args.decks)},
        "settings": {"seed": args.seed, "max_games": args.max_games,
                     "max_deck_rows": args.max_deck_rows, "max_turn": args.max_turn,
                     "basic_lands_included": True,
                     "zero_mulligans_only": not args.include_mulligans,
                     "allow_unknown_departures": args.allow_unknown_departures},
        "replay_rows": len(rows), "replay_events": len(groups),
        "held_out_events": len(test_groups), "held_out_games": len(seen_games),
        "training_decks": len(train),
        "training_event_overlap_with_test": len({d.group_id for d in train} & test_groups),
        "evaluated_games": len(used_games), "evaluated_snapshots": snapshots_count,
        "evaluated_games_by_mulligans": dict(sorted(evaluated_games_by_mulligans.items())),
        "stopped_games_by_reason": dict(stops),
        "unsupported_prior_snapshots": unsupported, "card_universe_size": len(universe),
        "aggregation": "average snapshots within game, then average games",
        **summary, "examples": examples,
        "limitations": ["File-prefix sample", "Half-turn summaries only; no within-turn action order",
                        "Labels are audited tracked-player hands, viewed from their opponent's perspective",
                        "Random departures ignore strategic card selection and passes",
                        "Uniform opening hands ignore Arena hand smoothing",
                        "When enabled, mulligan hands use a uniform retained-hand approximation; keep choices and bottoming order are ignored",
                        "Library changes from milling, scrying, or effects outside ordinary draws are ignored",
                        "Snapshots stop when movement accounting fails; mulligans are excluded unless explicitly enabled",
                        "Casting zone provenance is unavailable; matching accounting is only a consistency check",
                        "Draw counts and hand sizes are derived from lengths, never private card identities",
                        "No held-out decklists or private hand identities passed to the model",
                        "No-compatible-prior snapshots are excluded and counted"]}
    if args.allow_unknown_departures:
        report["limitations"].append(
            "Unidentified net losses use public counts only and are marginalized as random departures; causes and identities are unknown")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    print(f"Inspectable examples: {args.output}")


if __name__ == "__main__":
    main()
