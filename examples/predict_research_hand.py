"""Score one public-history JSON using locally trained research artifacts."""
import argparse
import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from mtg_opponent_model.hand_features import feature_matrix,HISTORY_FEATURES
from mtg_opponent_model.hands import PublicHistory,PublicStep


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='JSON containing history and public board only')
    parser.add_argument('--artifact',type=Path,default=Path('data/hand_research_models.joblib'))
    parser.add_argument('--model',choices=['boosted_history','boosted_board','smoothed_prior'],default='boosted_history')
    parser.add_argument('--calibrated',action='store_true')
    args=parser.parse_args()
    payload=json.loads(args.input.read_text())
    if set(payload)-{'history','board'}:
        raise ValueError('Input must contain only public history and board, not evaluation labels')
    public=payload['history']
    history=PublicHistory(public['opening_hand_size'],tuple(PublicStep(
        turn=s['turn'],active_player=s['active_player'],draws=s['draws'],
        departed=tuple(s['departed']),hand_size=s['hand_size'],
        unknown_departures=s.get('unknown_departures',0)) for s in public['steps']))
    artifact=joblib.load(args.artifact)
    with threadpool_limits(limits=4):
        probabilities=artifact['prior'].predict(history)
        if args.model!='smoothed_prior':
            include_board=args.model=='boosted_board'
            features=feature_matrix(artifact['cards'],artifact['metadata'],history,probabilities,
                                    payload.get('board',{}),include_board=include_board)
            values=artifact['models'][args.model].predict(features,calibrated=args.calibrated)
            probabilities=dict(zip(artifact['cards'],values.tolist()))
    print(json.dumps({'model':args.model,'calibrated':args.calibrated,
                      'top_20':sorted(probabilities.items(),key=lambda x:(-x[1],x[0]))[:20]},indent=2))


if __name__=='__main__':
    main()
