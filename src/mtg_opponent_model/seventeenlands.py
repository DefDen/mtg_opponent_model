"""Streaming adapter for 17Lands public game-data CSVs (plain or gzip)."""
import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .data import DeckRecord


@dataclass
class LoadStats:
    rows_read: int = 0
    duplicate_rows: int = 0
    incomplete_rows: int = 0
    decks_loaded: int = 0


class SeventeenLandsLoader:
    """Extract main decks, deduplicating within each draft by exact copy counts.

    max_rows limits game rows, not decks; a prefix is not a random sample.
    Decks with fewer than 40 cards are skipped as incomplete.
    """

    def __init__(self, path: Path, expansion: str, event_type: str,
                 max_rows: int | None = None) -> None:
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.path = path
        self.expansion = expansion
        self.event_type = event_type
        self.max_rows = max_rows
        self.stats = LoadStats()

    def load(self) -> Iterator[DeckRecord]:
        self.stats = LoadStats()
        seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            required = {"draft_id", "expansion", "event_type"}
            columns = sorted(c for c in fields if c.startswith("deck_"))
            if not required.issubset(fields) or not columns:
                raise ValueError("Missing 17Lands metadata or deck_ columns")
            for row in reader:
                if self.max_rows is not None and self.stats.rows_read >= self.max_rows:
                    break
                self.stats.rows_read += 1
                if (row["expansion"] != self.expansion
                        or row["event_type"] != self.event_type):
                    raise ValueError(f"Unexpected set/format on CSV line {reader.line_num}")
                draft_id = row["draft_id"]
                if not draft_id:
                    raise ValueError(f"Missing draft_id on CSV line {reader.line_num}")
                try:
                    counts = {column[5:]: int(row[column]) for column in columns}
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"Invalid deck count on CSV line {reader.line_num}") from exc
                if any(count < 0 for count in counts.values()):
                    raise ValueError(f"Negative deck count on CSV line {reader.line_num}")
                cards = {card: count for card, count in counts.items() if count}
                if sum(cards.values()) < 40:
                    self.stats.incomplete_rows += 1
                    continue
                signature = tuple(sorted(cards.items()))
                key = (draft_id, signature)
                if key in seen:
                    self.stats.duplicate_rows += 1
                    continue
                seen.add(key)
                digest = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
                self.stats.decks_loaded += 1
                yield DeckRecord(f"{draft_id}:{digest}", draft_id, cards,
                                 self.expansion, self.event_type)
