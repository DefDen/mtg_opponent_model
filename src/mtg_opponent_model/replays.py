"""17Lands replay labels kept separate from public hand-model inputs."""
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Mapping

from .hands import PublicHistory, PublicStep


@dataclass(frozen=True)
class HandSnapshot:
    game_id: str
    group_id: str
    public: PublicHistory
    actual_hand: Mapping[str, int]  # Evaluation only; never passed to the model.


def load_card_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "name"}.issubset(reader.fieldnames or []):
            raise ValueError("Card mapping needs id and name columns")
        return {row["id"]: row["name"] for row in reader}


def card_list(value: str, names: Mapping[str, str]) -> tuple[str, ...]:
    """Arena IDs are printing-specific; normalize them to card names."""
    return tuple(names[card] for card in value.split("|") if card)


def reconstruct_snapshots(row: Mapping[str, str], names: Mapping[str, str],
                          max_turn: int = 3) -> tuple[list[HandSnapshot], str | None]:
    """Validate early-turn hand accounting, stopping at unsupported movement.

    Private opening/drawn/hand identities are used only to reconstruct and
    audit labels. Public draw counts and hand sizes use list lengths only.
    With zero mulligans and ordinary movements, labels must match exactly.
    Summaries lack zone provenance; consistent accounting isn't proof that
    every cast came from hand. Retain this limitation in experiment reports.
    """
    if int(row["num_mulligans"]) != 0:
        return [], "mulligan"
    if row["on_play"] not in {"True", "False"}:
        raise ValueError("Invalid on_play")
    opening = card_list(row["opening_hand"], names)
    if len(opening) != 7:
        return [], "opening_hand_size"
    hand = Counter(opening)
    steps: list[PublicStep] = []
    snapshots = []
    game_id = ":".join(row[k] for k in ("draft_id", "match_number", "game_number"))
    sides = ("user", "oppo") if row["on_play"] == "True" else ("oppo", "user")
    for turn in range(1, max_turn + 1):
        for side in sides:
            prefix = f"{side}_turn_{turn}_"
            # Padded/nonexistent half-turns have both life values zero.
            if (float(row[prefix + "eot_user_life"] or 0) == 0
                    and float(row[prefix + "eot_oppo_life"] or 0) == 0):
                return snapshots, None
            draws = card_list(row.get(prefix + "cards_drawn", ""), names) if side == "user" else ()
            if side == "user" and row.get(prefix + "cards_tutored", ""):
                return snapshots, "tutor"
            keys = (["lands_played", "creatures_cast", "non_creatures_cast", "cards_discarded"]
                    if side == "user" else [])
            # This column records target instants/sorceries on either half-turn.
            keys.append("user_instants_sorceries_cast")
            departed = tuple(card for key in keys
                             for card in card_list(row.get(prefix + key, ""), names))
            next_hand = hand.copy()
            next_hand.update(draws)
            for card in departed:
                next_hand[card] -= 1
                if next_hand[card] < 0:
                    return snapshots, "departure_not_in_hand"
            next_hand = +next_hand
            actual = Counter(card_list(row[prefix + "eot_user_cards_in_hand"], names))
            if next_hand != actual:
                return snapshots, "unaccounted_hand_movement"
            # Only counts and public departure identities cross the boundary.
            steps.append(PublicStep(turn, "target" if side == "user" else "observer",
                                    len(draws), departed, sum(actual.values())))
            snapshots.append(HandSnapshot(game_id, row["draft_id"],
                                          PublicHistory(7, tuple(steps)), dict(actual)))
            hand = actual
    return snapshots, None


def read_replay_rows(path: Path, max_games: int = 500) -> list[dict[str, str]]:
    """Read complete game rows from a valid CSV/gzip file; no partial gzip input."""
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"draft_id", "match_number", "game_number", "opening_hand", "on_play",
                    "num_mulligans", "user_turn_1_eot_user_cards_in_hand"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing replay metadata or hand columns")
        rows = []
        for row in reader:
            if len(rows) >= max_games:
                break
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Incomplete replay CSV row")
            rows.append(row)
        return rows
