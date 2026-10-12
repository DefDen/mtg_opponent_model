"""Paired hand metrics with bootstrap uncertainty clustered by draft event."""
import random
from typing import Mapping, Sequence


def summarize_hand_scores(
    scores: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]],
    game_groups: Mapping[str, str], *, bootstrap_samples: int = 1000, seed: int = 42,
) -> dict:
    """Average snapshots within games; resample whole events for paired deltas.

    Each bootstrap draw samples the original number of draft events with
    replacement. All games in a sampled event travel together. Method pairs
    use identical draws. Deltas are public_movements minus no_history; lower
    Brier is better, so a negative Brier delta favors public movements.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    methods = ("public_movements", "no_history")
    if set(scores) != set(methods):
        raise ValueError("Need public_movements and no_history scores")
    games = sorted(scores[methods[0]])
    if not games or set(games) != set(scores[methods[1]]) or not set(games).issubset(game_groups):
        raise ValueError("Need nonempty paired games and their event groups")
    first = scores[methods[0]][games[0]]
    if not first or not first[0]:
        raise ValueError("Need snapshot metrics")
    keys = sorted(first[0])
    per_game = {method: {} for method in methods}
    for game in games:
        if len(scores[methods[0]][game]) != len(scores[methods[1]][game]):
            raise ValueError("Methods must use the same snapshots per game")
        for method in methods:
            rows = scores[method][game]
            if not rows or any(set(row) != set(keys) for row in rows):
                raise ValueError("Need consistent nonempty snapshot metrics")
            per_game[method][game] = {key: sum(row[key] for row in rows) / len(rows)
                                     for key in keys}
    means = {method: {key: sum(per_game[method][game][key] for game in games) / len(games)
                      for key in keys} for method in methods}
    event_games = {}
    for game in games:
        event_games.setdefault(game_groups[game], []).append(game)
    events = sorted(event_games)
    event_deltas = {event: {key: sum(per_game[methods[0]][game][key]
                                    - per_game[methods[1]][game][key]
                                    for game in event_games[event]) for key in keys}
                    for event in events}
    rng = random.Random(seed)
    draws = {key: [] for key in keys}
    for _ in range(bootstrap_samples):
        sampled = rng.choices(events, k=len(events))
        count = sum(len(event_games[event]) for event in sampled)
        for key in keys:
            draws[key].append(sum(event_deltas[event][key] for event in sampled) / count)

    def percentile(values, fraction):
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"metrics": means,
            "paired_differences": {key: {
                "estimate": means[methods[0]][key] - means[methods[1]][key],
                "ci95": [percentile(draws[key], 0.025), percentile(draws[key], 0.975)],
            } for key in keys},
            "uncertainty": {"method": "paired percentile bootstrap over draft events",
                            "samples": bootstrap_samples, "seed": seed,
                            "evaluated_events": len(events),
                            "difference": "public_movements minus no_history"}}
