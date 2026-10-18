"""The presentation adapter must only read public end-of-turn information."""
from website.scripts.board_data import public_board


def test_public_board_preserves_copies_without_reading_private_hand_ids():
    row = {
        'user_turn_2_eot_user_lands_in_play': '1|1',
        'user_turn_2_eot_user_creatures_in_play': '2',
        'user_turn_2_eot_user_non_creatures_in_play': '',
        'user_turn_2_eot_user_life': '17.0',
        'user_turn_2_eot_oppo_life': '20.0',
        'user_turn_2_eot_oppo_cards_in_hand': '6.0',
        'user_turn_2_eot_user_cards_in_hand': 'private-id-not-in-mapping',
    }
    result = public_board(row, {'1': 'Land', '2': 'Creature'}, 2, 'target')
    assert result['target'] == {'lands': ['Land', 'Land'], 'creatures': ['Creature'],
                                'other': [], 'life': 17}
    assert result['observer']['hand_size'] == 6
    assert 'actual_hand' not in result['target']


def test_board_reads_selected_half_turn_and_does_not_accumulate_removed_cards():
    row = {
        'oppo_turn_2_eot_user_lands_in_play': '1',
        'oppo_turn_2_eot_user_creatures_in_play': '',
        'oppo_turn_2_eot_user_non_creatures_in_play': '',
        'oppo_turn_2_eot_user_life': '16.0',
        'oppo_turn_2_eot_oppo_life': '19.0',
        'oppo_turn_2_eot_oppo_cards_in_hand': '5.0',
        'user_turn_2_eot_user_creatures_in_play': '2',
    }
    result = public_board(row, {'1': 'Land', '2': 'Creature'}, 2, 'observer')
    assert result['target']['creatures'] == []
    assert result['target']['life'] == 16


def test_full_game_coverage_uses_all_recorded_half_turns_not_three_round_limit():
    from website.scripts.export_full_game import recorded_steps

    row = {'on_play': 'True'}
    for turn in range(1, 10):
        for side in ('user', 'oppo'):
            row[f'{side}_turn_{turn}_eot_user_life'] = '6.0'
            row[f'{side}_turn_{turn}_eot_oppo_life'] = '8.0'
    # Last opponent turn and later rounds are padding, not recorded states.
    row['oppo_turn_9_eot_user_life'] = '0.0'
    row['oppo_turn_9_eot_oppo_life'] = '0.0'
    steps = recorded_steps(row)
    assert len(steps) == 17
    assert steps[0] == (1, 'target')
    assert steps[-1] == (9, 'target')
