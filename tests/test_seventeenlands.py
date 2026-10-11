import csv
import gzip

import pytest

from mtg_opponent_model.seventeenlands import SeventeenLandsLoader


FIELDS = ["expansion", "event_type", "draft_id", "deck_Plains", "deck_Creature",
          "sideboard_Removal", "drawn_Removal"]


def write_rows(path, rows, fields=FIELDS):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def row(draft="a", lands=17, creature=23):
    return ["BLB", "PremierDraft", draft, lands, creature, 3, 1]


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_preserve_counts_and_deduplicate_within_events(tmp_path, suffix):
    path = tmp_path / ("games" + suffix)
    write_rows(path, [row(), row(), row(creature=24), row("b"), row("c", 0, 0)])
    loader = SeventeenLandsLoader(path, "BLB", "PremierDraft")
    decks = list(loader.load())
    assert len(decks) == 3
    assert decks[0].cards == {"Creature": 23, "Plains": 17}
    assert decks[0].group_id == decks[1].group_id
    assert decks[0].deck_id != decks[1].deck_id
    assert decks[2].group_id == "b"
    assert loader.stats.rows_read == 5
    assert loader.stats.duplicate_rows == 1
    assert loader.stats.incomplete_rows == 1
    assert list(loader.load()) == decks
    assert loader.stats.rows_read == 5


def test_row_limit(tmp_path):
    path = tmp_path / "games.csv"
    write_rows(path, [row(), row(), row("b")])
    loader = SeventeenLandsLoader(path, "BLB", "PremierDraft", max_rows=2)
    assert len(list(loader.load())) == 1
    assert loader.stats.rows_read == 2


@pytest.mark.parametrize("bad_row", [row(lands=-1), row(lands=""), row(lands="1.5"), row(draft="")])
def test_malformed_rows_fail(tmp_path, bad_row):
    path = tmp_path / "games.csv"
    write_rows(path, [bad_row])
    with pytest.raises(ValueError):
        list(SeventeenLandsLoader(path, "BLB", "PremierDraft").load())


def test_missing_schema_and_wrong_format(tmp_path):
    path = tmp_path / "games.csv"
    write_rows(path, [], fields=["draft_id"])
    with pytest.raises(ValueError):
        list(SeventeenLandsLoader(path, "BLB", "PremierDraft").load())
    write_rows(path, [row()])
    with pytest.raises(ValueError):
        list(SeventeenLandsLoader(path, "BLB", "Sealed").load())
