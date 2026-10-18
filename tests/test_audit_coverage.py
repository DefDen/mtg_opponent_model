from examples.audit_replays import audit


def row_with_unidentified_loss():
    row = {'draft_id': 'g', 'match_number': '0', 'game_number': '0',
           'on_play': 'True', 'num_mulligans': '0', 'num_turns': '1',
           'opening_hand': '1|1|1|1|2|2|2'}
    for side in ('user', 'oppo'):
        row[f'{side}_turn_1_eot_user_life'] = '20'
        row[f'{side}_turn_1_eot_oppo_life'] = '20'
        row[f'{side}_turn_1_eot_user_cards_in_hand'] = row['opening_hand']
    row['oppo_turn_1_eot_user_cards_in_hand'] = '1|1|1|2|2|2'
    return row


def test_coverage_distinguishes_recorded_states_from_audited_states():
    row = row_with_unidentified_loss()
    strict = audit([row, row], {'1': 'A', '2': 'B'}, 1)
    expanded = audit([row, row], {'1': 'A', '2': 'B'}, 1, allow_unknown_departures=True)
    assert strict['unique_games'] == expanded['unique_games'] == 1
    assert strict['available_recorded_snapshots_in_window'] == 2
    assert strict['valid_snapshots'] == 1
    assert strict['games_reaching_recorded_end'] == 0
    assert expanded['valid_snapshots'] == 2
    assert expanded['games_reaching_recorded_end'] == 1
    assert expanded['snapshots_with_unknown_departures'] == 1
    assert expanded['unknown_departure_cards'] == 1


def test_short_window_does_not_count_an_entire_game_as_complete():
    row = row_with_unidentified_loss()
    row['num_turns'] = '2'
    for side in ('user', 'oppo'):
        row[f'{side}_turn_2_eot_user_life'] = '20'
        row[f'{side}_turn_2_eot_oppo_life'] = '20'
    result = audit([row], {'1': 'A', '2': 'B'}, 1, allow_unknown_departures=True)
    assert result['games_without_reconstruction_stop'] == 1
    assert result['games_reaching_recorded_end'] == 0
