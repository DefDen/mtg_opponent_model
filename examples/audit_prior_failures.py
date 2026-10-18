"""Diagnose the earlier finite-deck prior failures on every held-out snapshot."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from mtg_opponent_model.priors import IndexedDeckPrior, history_evidence
from mtg_opponent_model.replays import load_card_names, read_replay_rows, reconstruct_snapshots
from mtg_opponent_model.research_data import split_events
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader


def main() -> None:
    path = Path('data/replay_sample_5000.BLB.PremierDraft.csv.gz')
    rows = read_replay_rows(path, 5000)
    names = load_card_names(Path('data/cards.csv'))
    split = split_events((row['draft_id'] for row in rows))
    decks = [d for d in SeventeenLandsLoader(Path('data/game_data_public.BLB.PremierDraft.csv.gz'),
                                            'BLB', 'PremierDraft', 20000).load()
             if d.group_id not in split['test']]
    universe = sorted(key[5:] for key in rows[0] if key.startswith('deck_'))
    prior = IndexedDeckPrior(decks, universe)
    reasons = Counter()
    overlap = Counter()
    examples = []
    games = set()
    for row in rows:
        if row['draft_id'] not in split['test']:
            continue
        game = ':'.join(row[k] for k in ('draft_id', 'match_number', 'game_number'))
        if game in games:
            continue
        games.add(game)
        snapshots, _ = reconstruct_snapshots(row, names, 30, allow_mulligans=True,
                                            allow_unknown_departures=True)
        recorded = {key[5:]: int(value) for key, value in row.items() if key.startswith('deck_') and int(value)}
        for snapshot in snapshots:
            diagnosis = prior.diagnose(snapshot.public, recorded)
            reasons[diagnosis['reason']] += 1
            overlap[f"{'supported' if diagnosis['compatible_decks'] else 'unsupported'}:recorded_pool_violation={diagnosis['recorded_pool_violation_evaluation_only']}"] += 1
            if diagnosis['reason'] != 'supported' and len(examples) < 12:
                examples.append({'game': game, 'turn': snapshot.public.steps[-1].turn,
                                 'public_departures': dict(history_evidence(snapshot.public).counts),
                                 **diagnosis})
    report = {'scope': 'Reproduce all 12,448 earlier audited held-out states with the original test-only excluded prior',
              'training_decks': len(decks), 'held_out_games': len(games),
              'audited_snapshots': sum(reasons.values()), 'reasons': dict(reasons),
              'support_and_accounting': dict(overlap), 'examples_evaluation_only': examples,
              'split_seed': 42, 'source': '17Lands public data, CC BY 4.0',
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  path, Path('data/cards.csv'), Path('data/game_data_public.BLB.PremierDraft.csv.gz'))},
              'limitations': ['Recorded deck comparisons are evaluation-only, never inference inputs',
                              'Exceeding a recorded copy count flags possible unmodeled reuse/generated cards; it does not establish the cause',
                              'This audit retains the earlier split to reproduce the previous failures; the new tree experiment also excludes validation events']}
    output = Path('reports/blb_prior_failure_audit.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k,v in report.items() if k not in ('examples_evaluation_only','input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
