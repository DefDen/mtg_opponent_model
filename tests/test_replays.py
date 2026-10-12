from mtg_opponent_model.replays import extract_public_history, reconstruct_snapshots
import pytest


NAMES = {"1": "A", "2": "B", "3": "C"}


def make_row(on_play="True"):
    row = {"draft_id": "draft", "match_number": "0", "game_number": "0",
           "num_mulligans": "0", "on_play": on_play,
           "opening_hand": "1|1|1|1|2|2|2"}
    for side in ("user", "oppo"):
        prefix = side + "_turn_1_"
        row[prefix + "eot_user_life"] = "20.0"
        row[prefix + "eot_oppo_life"] = "20.0"
        row[prefix + "eot_user_cards_in_hand"] = "1|1|1|1|2|2|2"
    return row


def test_public_inputs_never_contain_private_hand_or_draw_identity():
    first = make_row()
    first["user_turn_1_cards_drawn"] = "3"
    first["user_turn_1_eot_user_cards_in_hand"] += "|3"
    first["oppo_turn_1_eot_user_cards_in_hand"] += "|3"
    a, reason = reconstruct_snapshots(first, NAMES, 1)
    assert reason is None
    second = dict(first)
    second["user_turn_1_cards_drawn"] = "2"
    second["user_turn_1_eot_user_cards_in_hand"] = "1|1|1|1|2|2|2|2"
    second["oppo_turn_1_eot_user_cards_in_hand"] = second["user_turn_1_eot_user_cards_in_hand"]
    b, reason = reconstruct_snapshots(second, NAMES, 1)
    assert reason is None
    assert [s.public for s in a] == [s.public for s in b]
    assert a[0].actual_hand != b[0].actual_hand


def test_turn_order_on_draw_and_public_departures():
    row = make_row("False")
    row["user_turn_1_cards_drawn"] = "2"
    row["user_turn_1_lands_played"] = "1"
    row["user_turn_1_eot_user_cards_in_hand"] = "1|1|1|2|2|2|2"
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1)
    assert reason is None
    assert snapshots[0].public.steps[-1].active_player == "observer"
    assert snapshots[1].public.steps[-1].departed == ("A",)
    assert snapshots[1].actual_hand == {"A": 3, "B": 4}


def test_observer_turn_target_spell_is_accounted_for():
    row = make_row()
    row["oppo_turn_1_user_instants_sorceries_cast"] = "2"
    row["oppo_turn_1_eot_user_cards_in_hand"] = "1|1|1|1|2|2"
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1)
    assert reason is None
    assert snapshots[-1].public.steps[-1].departed == ("B",)


def test_stop_before_unaccounted_or_tutored_movement():
    row = make_row()
    row["oppo_turn_1_eot_user_cards_in_hand"] += "|3"
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1)
    assert len(snapshots) == 1
    assert reason == "unaccounted_hand_movement"
    row["user_turn_1_cards_tutored"] = "3"
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1)
    assert snapshots == []
    assert reason == "tutor"
    row["num_mulligans"] = "1"
    assert reconstruct_snapshots(row, NAMES, 1) == ([], "mulligan")


def test_stop_diagnostics_are_evaluation_only_and_preserve_snapshots():
    row = make_row()
    row["oppo_turn_1_eot_user_cards_in_hand"] += "|3"
    details = {"stale": True}
    expected = reconstruct_snapshots(row, NAMES, 1)
    assert reconstruct_snapshots(row, NAMES, 1, diagnostics=details) == expected
    assert details == {"turn": 1, "active_player": "observer",
                       "added_cards_evaluation_only": {"C": 1},
                       "missing_cards_evaluation_only": {}}
    assert reconstruct_snapshots(make_row(), NAMES, 1, diagnostics=details)[1] is None
    assert details == {}


def test_diagnostics_clear_at_game_end_and_early_exclusion():
    row = make_row()
    row["oppo_turn_1_eot_user_life"] = "0"
    row["oppo_turn_1_eot_oppo_life"] = "0"
    details = {"stale": True}
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1, diagnostics=details)
    assert len(snapshots) == 1 and reason is None and details == {}
    row["num_mulligans"] = "1"
    details["stale"] = True
    assert reconstruct_snapshots(row, NAMES, 1, diagnostics=details) == ([], "mulligan")
    assert details == {}


@pytest.mark.parametrize("max_turn", [0, 31])
def test_invalid_turn_limit(max_turn):
    with pytest.raises(ValueError, match="max_turn"):
        reconstruct_snapshots(make_row(), NAMES, max_turn)


def test_public_adapter_does_not_resolve_private_ids():
    row = make_row()
    row["opening_hand"] = "unknown|unknown|unknown|unknown|unknown|unknown|unknown"
    row["user_turn_1_cards_drawn"] = "private-draw"
    row["user_turn_1_lands_played"] = "1"
    row["user_turn_1_eot_user_cards_in_hand"] = row["opening_hand"]
    row["oppo_turn_1_eot_user_cards_in_hand"] = row["opening_hand"]
    history, reason = extract_public_history(row, NAMES, 1)
    assert reason is None
    assert history.steps[0].draws == 1
    assert history.steps[0].departed == ("A",)
    assert history.steps[0].hand_size == 7


def test_public_count_consistency_does_not_replace_identity_audit():
    row = make_row()
    row["oppo_turn_1_eot_user_cards_in_hand"] = "1|1|1|1|2|2|3"
    history, reason = extract_public_history(row, NAMES, 1)
    assert reason is None and len(history.steps) == 2
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1)
    assert reason == "unaccounted_hand_movement" and len(snapshots) == 1


def test_observer_movements_never_become_target_draws_or_departures():
    row = make_row()
    expected = extract_public_history(row, NAMES, 1)
    row.update(oppo_turn_1_cards_drawn_or_tutored="4",
               oppo_turn_1_cards_discarded="unmapped-observer-card",
               oppo_turn_1_creatures_cast="unmapped-observer-creature")
    assert extract_public_history(row, NAMES, 1) == expected
    row["oppo_turn_1_eot_user_cards_in_hand"] += "|3"
    history, reason = extract_public_history(row, NAMES, 1)
    assert reason == "unaccounted_hand_size" and len(history.steps) == 1


def test_public_history_matches_audited_inputs_and_stops_at_tutor():
    row = make_row()
    history, reason = extract_public_history(row, NAMES, 1)
    snapshots, audit_reason = reconstruct_snapshots(row, NAMES, 1)
    assert reason is audit_reason is None
    assert history == snapshots[-1].public
    row["user_turn_1_cards_tutored"] = "private-tutor"
    history, reason = extract_public_history(row, NAMES, 1)
    assert reason == "tutor" and history.steps == ()


@pytest.mark.parametrize("mulligans", [1, 2, 7])
def test_opt_in_mulligan_support_uses_kept_hand_size(mulligans):
    row = make_row("False")
    row["num_mulligans"] = str(mulligans)
    kept = "|".join(["1"] * (7 - mulligans))
    row["opening_hand"] = kept
    row["oppo_turn_1_eot_user_cards_in_hand"] = kept
    row["user_turn_1_cards_drawn"] = "2"
    row["user_turn_1_eot_user_cards_in_hand"] = "|".join(filter(None, [kept, "2"]))
    assert reconstruct_snapshots(row, NAMES, 1) == ([], "mulligan")
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1, allow_mulligans=True)
    public, public_reason = extract_public_history(row, NAMES, 1, allow_mulligans=True)
    assert reason is public_reason is None
    assert len(snapshots) == 2
    assert public == snapshots[-1].public
    assert public.opening_hand_size == 7 - mulligans
    assert public.steps[-1].hand_size == 8 - mulligans


def test_mulligan_kept_hand_still_requires_exact_identity_accounting():
    row = make_row()
    row["num_mulligans"] = "1"
    row["opening_hand"] = "1|1|1|1|2|2"
    row["user_turn_1_eot_user_cards_in_hand"] = "1|1|1|1|2|3"
    snapshots, reason = reconstruct_snapshots(row, NAMES, 1, allow_mulligans=True)
    assert snapshots == [] and reason == "unaccounted_hand_movement"
    row["opening_hand"] += "|2"
    assert reconstruct_snapshots(row, NAMES, 1, allow_mulligans=True) == ([], "opening_hand_size")


@pytest.mark.parametrize("mulligans", [-1, 8])
def test_reject_invalid_mulligan_counts(mulligans):
    row = make_row()
    row["num_mulligans"] = str(mulligans)
    with pytest.raises(ValueError, match="num_mulligans"):
        extract_public_history(row, NAMES, 1, allow_mulligans=True)
