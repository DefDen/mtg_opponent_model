"""Public battlefield metadata, isolated from prediction and private-hand labels."""
from typing import Mapping


def public_board(row: Mapping[str, str], names: Mapping[str, str], turn: int,
                 active_player: str) -> dict:
    """Read end-of-half-turn public zones, never private hand identities.

    Tapping, damage, counters, attachments, and within-turn ordering are not
    represented by these card-type lists. Duplicate names retain copy counts.
    """
    side = "user" if active_player == "target" else "oppo"
    prefix = f"{side}_turn_{turn}_"

    def cards(field: str) -> list[str]:
        return [names[card] for card in row.get(prefix + field, "").split("|") if card]

    result = {}
    for label, key in (("target", "user"), ("observer", "oppo")):
        player = {"lands": cards(f"eot_{key}_lands_in_play"),
                  "creatures": cards(f"eot_{key}_creatures_in_play"),
                  "other": cards(f"eot_{key}_non_creatures_in_play"),
                  "life": float(row[prefix + f"eot_{key}_life"])}
        if label == "observer":
            # Public count only. No observer hand identities are available.
            player["hand_size"] = int(float(row[prefix + "eot_oppo_cards_in_hand"] or 0))
        result[label] = player
    return result
