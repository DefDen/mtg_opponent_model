"""Add public-input research predictions to existing held-out replay exports.

Run after export_data.py and export_full_game.py. Uses trusted local joblib only;
no model artifact or private labels are shipped as inference inputs to the site.
"""
import hashlib
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from mtg_opponent_model.hand_features import feature_matrix
from mtg_opponent_model.hands import PublicHistory, PublicStep

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    'historical': {'label': 'Historical decks', 'description': 'Interpretable finite-deck hand accounting. Uses 4,180 historical decks.'},
    'hand_size': {'label': 'Hand size only', 'description': 'Historical deck frequencies and current hand size; ignores revealed cards.'},
    'smoothed_prior': {'label': 'Smoothed prior', 'description': 'Color and deck-shape mixture allowing unseen card combinations. Uses 4,096 training decks.'},
    'boosted_history': {'label': 'Trees · public history', 'description': 'Boosted trees using public hand movements, card metadata, and the smoothed prior. Raw probabilities.'},
    'boosted_board': {'label': 'Trees · history + board', 'description': 'Boosted trees also using public board counts, life totals, and land colors. Raw probabilities.'},
}


def predict_methods(artifact: dict, history: PublicHistory, board: dict) -> dict:
    """Accept only public state. Keep private labels out of feature generation."""
    prior = artifact['prior'].predict(history)
    result = {'smoothed_prior': prior}
    for method in ('boosted_history', 'boosted_board'):
        features = feature_matrix(artifact['cards'], artifact['metadata'], history,
                                  prior, board, include_board=method == 'boosted_board')
        values = artifact['models'][method].predict(features, calibrated=False)
        result[method] = dict(zip(artifact['cards'], values.tolist()))
    return result


def main() -> None:
    path = ROOT / 'website/dist/data.json'
    dataset = json.loads(path.read_text())
    artifact_path = ROOT / 'data/hand_research_models.joblib'
    report = json.loads((ROOT / 'reports/blb_model_comparison.json').read_text())
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != report['artifacts']['sha256']:
        raise ValueError('Model artifact does not match the recorded experiment')
    artifact = joblib.load(artifact_path)
    split = json.loads((ROOT / 'data/hand_research_split.json').read_text())
    with threadpool_limits(limits=4):
        for game in dataset['cases']:
            if game['source_game'].split(':')[0] not in split['test']:
                raise ValueError('Replay is not in the research test events')
            steps = []
            for frame in game['frames']:
                step = dict(frame['step'])
                step['departed'] = tuple(step['departed'])
                steps.append(PublicStep(**step))
                history = PublicHistory(game['opening_hand_size'], tuple(steps))
                results = predict_methods(artifact, history, frame['board'])
                frame['methods'] = {'historical': frame['probabilities'], 'hand_size': frame['baseline'], **results}
    dataset['methods'] = METHODS
    dataset['method_comparison_note'] = (
        'Selected replay demonstrations, not a new benchmark. All replay draft events are held out. '
        'Methods use different training pools and assumptions; boosted trees use raw, uncalibrated probabilities. '
        'Research models were evaluated on sampled states through turn 30; these displayed games are selected examples.')
    dataset['research_artifact_sha256'] = digest
    path.write_text(json.dumps(dataset, separators=(',', ':')))
    print('Exported five methods for', sum(len(g['frames']) for g in dataset['cases']), 'held-out states')


if __name__ == '__main__':
    main()
