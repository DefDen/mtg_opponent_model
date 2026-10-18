"""Event-disjoint research datasets and public-only battlefield extraction."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Mapping

from .hands import PublicHistory
from .replays import card_list, reconstruct_snapshots


@dataclass(frozen=True)
class ResearchCase:
    game_id: str
    group_id: str
    history: PublicHistory
    board: dict
    actual_hand: Mapping[str, int]  # Label only.
    recorded_deck: Mapping[str, int]  # Evaluation-only diagnostics.


def split_events(groups: Iterable[str], seed: int = 42) -> dict[str, set[str]]:
    ordered = sorted(set(groups))
    if len(ordered) < 5:
        raise ValueError("Need at least five events for train/validation/test")
    random.Random(seed).shuffle(ordered)
    count = max(1, round(0.2 * len(ordered)))
    return {"test": set(ordered[:count]), "validation": set(ordered[count:2 * count]),
            "train": set(ordered[2 * count:])}


def public_board(row: Mapping[str, str], names: Mapping[str, str], history: PublicHistory) -> dict:
    step = history.steps[-1]
    side = "user" if step.active_player == "target" else "oppo"
    prefix = f"{side}_turn_{step.turn}_"
    result = {}
    for player, key in (("target", "user"), ("observer", "oppo")):
        result[player] = {"life": float(row[prefix + f"eot_{key}_life"] or 0),
                          "lands": card_list(row.get(prefix + f"eot_{key}_lands_in_play", ""), names),
                          "creatures": card_list(row.get(prefix + f"eot_{key}_creatures_in_play", ""), names),
                          "other": card_list(row.get(prefix + f"eot_{key}_non_creatures_in_play", ""), names)}
    return result


def make_cases(rows: Iterable[dict[str, str]], names: Mapping[str, str],
               seed: int = 42) -> tuple[list[ResearchCase], dict]:
    """One seeded random audited state per game, never chosen for model success.

    Audits include mulligan kept-hand approximation and unidentified losses.
    Copies of recorded decks stay confined to evaluation diagnostics.
    """
    cases = []
    seen = set()
    stops: Counter[str] = Counter()
    snapshot_count = 0
    for row in rows:
        game = ":".join(row[k] for k in ("draft_id", "match_number", "game_number"))
        if game in seen:
            continue
        seen.add(game)
        snapshots, reason = reconstruct_snapshots(row, names, 30, allow_mulligans=True,
                                                 allow_unknown_departures=True)
        if reason:
            stops[reason] += 1
        snapshot_count += len(snapshots)
        if not snapshots:
            continue
        rng = random.Random(int(hashlib.sha256(f"{seed}:{game}".encode()).hexdigest()[:16], 16))
        snapshot = snapshots[rng.randrange(len(snapshots))]
        actual_deck = {key[5:]: int(value) for key, value in row.items()
                       if key.startswith("deck_") and int(value) > 0}
        cases.append(ResearchCase(game, row["draft_id"], snapshot.public,
                                  public_board(row, names, snapshot.public),
                                  dict(snapshot.actual_hand), actual_deck))
    return cases, {"games": len(seen), "audited_snapshots": snapshot_count,
                   "games_with_cases": len(cases), "stops": dict(stops)}
