import pytest

from mtg_opponent_model.hand_evaluation import summarize_hand_scores


def test_equal_game_weight_and_paired_event_bootstrap():
    scores = {
        "public_movements": {"a": [{"Recall@5": 0}, {"Recall@5": 1}],
                             "b": [{"Recall@5": 1}], "c": [{"Recall@5": 1}]},
        "no_history": {"a": [{"Recall@5": 0}, {"Recall@5": 0}],
                       "b": [{"Recall@5": 0}], "c": [{"Recall@5": 0}]},
    }
    groups = {"a": "event1", "b": "event2", "c": "event2"}
    result = summarize_hand_scores(scores, groups)
    assert result["metrics"]["public_movements"]["Recall@5"] == pytest.approx(5 / 6)
    assert result["paired_differences"]["Recall@5"]["ci95"] == [0.5, 1.0]
    assert result["uncertainty"]["evaluated_events"] == 2
    assert summarize_hand_scores(scores, groups) == result


def test_paired_bootstrap_preserves_constant_difference():
    scores = {"public_movements": {"a": [{"Brier": 0.25}], "b": [{"Brier": 0.5}]},
              "no_history": {"a": [{"Brier": 0.5}], "b": [{"Brier": 0.75}]}}
    result = summarize_hand_scores(scores, {"a": "e1", "b": "e2"}, bootstrap_samples=20)
    assert result["paired_differences"]["Brier"] == {"estimate": -0.25, "ci95": [-0.25, -0.25]}


def test_bootstrap_keeps_correlated_games_in_each_event_together():
    scores = {"public_movements": {"a": [{"MRR": 0}], "b": [{"MRR": 1}],
                                  "c": [{"MRR": 0.5}], "d": [{"MRR": 0.5}]},
              "no_history": {game: [{"MRR": 0}] for game in "abcd"}}
    result = summarize_hand_scores(scores, {"a": "e1", "b": "e1", "c": "e2", "d": "e2"})
    # Both event means are 0.5 although their individual game scores differ.
    # Resampling games independently would introduce spurious variation.
    assert result["paired_differences"]["MRR"] == {"estimate": 0.5, "ci95": [0.5, 0.5]}


def test_reject_unpaired_games_or_snapshots():
    scores = {"public_movements": {"a": [{"Brier": 0.5}]},
              "no_history": {"b": [{"Brier": 0.5}]}}
    with pytest.raises(ValueError, match="paired"):
        summarize_hand_scores(scores, {"a": "e1", "b": "e2"})
    scores["no_history"] = {"a": [{"Brier": 0.5}, {"Brier": 0.5}]}
    with pytest.raises(ValueError, match="same snapshots"):
        summarize_hand_scores(scores, {"a": "e1"})


def test_reject_missing_groups_empty_scores_and_bad_sample_count():
    scores = {"public_movements": {"a": [{"Brier": 0.5}]},
              "no_history": {"a": [{"Brier": 0.5}]}}
    with pytest.raises(ValueError, match="groups"):
        summarize_hand_scores(scores, {})
    with pytest.raises(ValueError, match="positive"):
        summarize_hand_scores(scores, {"a": "e1"}, bootstrap_samples=0)
    with pytest.raises(ValueError, match="nonempty"):
        summarize_hand_scores({"public_movements": {}, "no_history": {}}, {})
