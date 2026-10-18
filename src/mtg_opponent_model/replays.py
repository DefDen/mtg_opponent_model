"""17Lands replay labels kept separate from public hand-model inputs."""
from collections import Counter
import csv
from dataclasses import dataclass, replace
import gzip
from pathlib import Path
from typing import Iterator, Mapping

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


def _card_count(value: str) -> int:
    """Count opaque IDs without resolving private identities."""
    return sum(bool(card) for card in value.split("|"))


def _account_for_loss(step: PublicStep, previous_size: int, enabled: bool) -> PublicStep:
    """Infer a loss COUNT from public sizes; never assign a cause or identity.

    Net gains and same-size replacements are not repaired. The independent
    audit must verify that the actual hand only lost cards before accepting.
    """
    deficit = previous_size + step.draws - len(step.departed) - step.hand_size
    if enabled and deficit > 0:
        return replace(step, unknown_departures=deficit)
    return step


def _validate_start(row: Mapping[str, str], max_turn: int, allow_mulligans: bool) -> str | None:
    if not 1 <= max_turn <= 30:
        raise ValueError("max_turn must be between 1 and 30")
    mulligans = int(row["num_mulligans"])
    if not 0 <= mulligans <= 7:
        raise ValueError("num_mulligans must be between 0 and 7")
    if mulligans and not allow_mulligans:
        return "mulligan"
    if row["on_play"] not in {"True", "False"}:
        raise ValueError("Invalid on_play")
    if _card_count(row["opening_hand"]) != 7 - mulligans:
        return "opening_hand_size"
    return None


class _TutorStop(Exception):
    def __init__(self, turn: int) -> None:
        self.turn = turn


def _public_steps(row: Mapping[str, str], names: Mapping[str, str],
                  max_turn: int) -> Iterator[tuple[str, PublicStep]]:
    """Read public identities and opaque private-list lengths only.

    Unprefixed movement fields describe the active player. In particular,
    oppo_turn_N_cards_drawn_or_tutored is an observer count, not a target draw.
    Missing target movements are never inferred from changes in private hands.
    """
    sides = ("user", "oppo") if row["on_play"] == "True" else ("oppo", "user")
    for turn in range(1, max_turn + 1):
        for side in sides:
            prefix = f"{side}_turn_{turn}_"
            if (float(row[prefix + "eot_user_life"] or 0) == 0
                    and float(row[prefix + "eot_oppo_life"] or 0) == 0):
                return
            if side == "user" and row.get(prefix + "cards_tutored", ""):
                raise _TutorStop(turn)
            draws = _card_count(row.get(prefix + "cards_drawn", "")) if side == "user" else 0
            keys = (["lands_played", "creatures_cast", "non_creatures_cast", "cards_discarded"]
                    if side == "user" else [])
            keys.append("user_instants_sorceries_cast")
            departed = tuple(card for key in keys
                             for card in card_list(row.get(prefix + key, ""), names))
            yield prefix, PublicStep(turn, "target" if side == "user" else "observer",
                                     draws, departed,
                                     _card_count(row[prefix + "eot_user_cards_in_hand"]))


def extract_public_history(row: Mapping[str, str], names: Mapping[str, str],
                           max_turn: int = 3, *,
                           allow_mulligans: bool = False,
                           allow_unknown_departures: bool = False) -> tuple[PublicHistory, str | None]:
    """Extract the longest count-consistent history without reading private identities.

    This is a public-input adapter, not an audited evaluation label. Cast source
    zones remain unknown. A count-consistent history may fail the independent
    identity audit in reconstruct_snapshots. Unknown private IDs are permitted.
    """
    reason = _validate_start(row, max_turn, allow_mulligans)
    history = PublicHistory(_card_count(row["opening_hand"]), ())
    if reason:
        return history, reason
    steps = []
    size = history.opening_hand_size
    try:
        for _, step in _public_steps(row, names, max_turn):
            step = _account_for_loss(step, size, allow_unknown_departures)
            if size + step.draws - len(step.departed) - step.unknown_departures != step.hand_size:
                return PublicHistory(history.opening_hand_size, tuple(steps)), "unaccounted_hand_size"
            steps.append(step)
            size = step.hand_size
    except _TutorStop:
        return PublicHistory(history.opening_hand_size, tuple(steps)), "tutor"
    return PublicHistory(history.opening_hand_size, tuple(steps)), None


def reconstruct_snapshots(row: Mapping[str, str], names: Mapping[str, str],
                          max_turn: int = 3, *,
                          allow_mulligans: bool = False,
                          allow_unknown_departures: bool = False,
                          diagnostics: dict | None = None) -> tuple[list[HandSnapshot], str | None]:
    """Independently audit public steps against private hand identities.

    Retain only the prefix passing exact identity accounting. Diagnostic label
    differences are evaluation-only and never supply missing public movements.
    Consistent accounting does not establish casting zone provenance.
    Opt-in mulligan support audits the kept hand of size 7 - num_mulligans;
    keep decisions and library-bottom order are not represented in the model.
    Opt-in unidentified departures expose only counts obtained independently
    from public hand sizes. Actual hands must be subsets of the accounted
    hand with exactly that many cards removed. No missing draw, return, or
    lost-card identity is copied from private labels into model inputs.
    """
    if diagnostics is not None:
        diagnostics.clear()
    reason = _validate_start(row, max_turn, allow_mulligans)
    if reason:
        return [], reason
    hand = Counter(card_list(row["opening_hand"], names))
    opening_size = sum(hand.values())
    steps: list[PublicStep] = []
    snapshots = []
    game_id = ":".join(row[k] for k in ("draft_id", "match_number", "game_number"))
    try:
        for prefix, step in _public_steps(row, names, max_turn):
            step = _account_for_loss(step, sum(hand.values()), allow_unknown_departures)
            if diagnostics is not None:
                diagnostics.clear()
                diagnostics.update(turn=step.turn, active_player=step.active_player)
            draws = (card_list(row.get(prefix + "cards_drawn", ""), names)
                     if step.active_player == "target" else ())
            next_hand = hand.copy()
            next_hand.update(draws)
            for card in step.departed:
                next_hand[card] -= 1
                if next_hand[card] < 0:
                    if diagnostics is not None:
                        diagnostics["departure_card_evaluation_only"] = card
                    return snapshots, "departure_not_in_hand"
            next_hand = +next_hand
            actual = Counter(card_list(row[prefix + "eot_user_cards_in_hand"], names))
            added = actual - next_hand
            missing = next_hand - actual
            if added or sum(missing.values()) != step.unknown_departures:
                if diagnostics is not None:
                    diagnostics.update(
                        added_cards_evaluation_only=dict(actual - next_hand),
                        missing_cards_evaluation_only=dict(next_hand - actual),
                    )
                return snapshots, "unaccounted_hand_movement"
            steps.append(step)
            snapshots.append(HandSnapshot(game_id, row["draft_id"],
                                          PublicHistory(opening_size, tuple(steps)), dict(actual)))
            hand = actual
    except _TutorStop as stop:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(turn=stop.turn, active_player="target")
        return snapshots, "tutor"
    if diagnostics is not None:
        diagnostics.clear()
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
