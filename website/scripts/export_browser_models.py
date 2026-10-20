"""Export trusted local fitted models as portable, read-only inference tables.

Run before publishing browser replay imports. No replay labels are exported.
"""
import hashlib
import json
from pathlib import Path
import random

import joblib

from mtg_opponent_model.replays import load_card_names, read_replay_rows
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    artifact_path = ROOT / 'data/hand_research_models.joblib'
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = json.loads((ROOT / 'reports/blb_model_comparison.json').read_text())
    if digest != report['artifacts']['sha256']:
        raise ValueError('Artifact checksum differs from research report')
    artifact = joblib.load(artifact_path)
    prior = artifact['prior']
    dataset = json.loads((ROOT / 'website/dist/data.json').read_text())
    settings = dataset['experiments'][1]['settings']
    rows = read_replay_rows(ROOT / 'data/replay_sample_5000.BLB.PremierDraft.csv.gz', 5000)
    groups = sorted({row['draft_id'] for row in rows})
    random.Random(settings['seed']).shuffle(groups)
    held_out = set(groups[:max(1, round(.2 * len(groups)))])
    decks = [d for d in SeventeenLandsLoader(ROOT / 'data/game_data_public.BLB.PremierDraft.csv.gz',
             'BLB', 'PremierDraft', settings['max_deck_rows']).load() if d.group_id not in held_out]
    assert len(decks) == dataset['training_decks']
    index = {c:i for i,c in enumerate(artifact['cards'])}
    trees = {}
    for name, model in artifact['models'].items():
        trees[name] = {
            'base': float(model.estimator._baseline_prediction[0,0]),
            'trees': [[[float(n['value']), int(n['feature_idx']), float(n['num_threshold']),
                        int(n['left']), int(n['right']), bool(n['is_leaf']), bool(n['missing_go_to_left'])]
                       for n in iteration[0].nodes] for iteration in model.estimator._predictors]}
    payload = {'schema_version':1, 'expansion':'BLB', 'artifact_sha256':digest,
               'cards':list(artifact['cards']),
               'metadata':{c:{'colors':m.colors,'types':m.types,'mana_value':m.mana_value}
                           for c,m in artifact['metadata'].items()},
               'arena_names':load_card_names(ROOT / 'data/cards.csv'),
               'land':prior.land.tolist(), 'alpha':{g:a.tolist() for g,a in prior.alpha.items()},
               'components':prior.components, 'trees':trees,
               'historical_decks':[[[index[c],n] for c,n in sorted(d.cards.items())] for d in decks]}
    dest = ROOT / 'website/dist/browser-models.json'
    dest.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
    print(f'Exported {len(decks)} decks and two fitted tree models: {dest.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
