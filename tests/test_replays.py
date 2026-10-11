from mtg_opponent_model.replays import reconstruct_snapshots


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
