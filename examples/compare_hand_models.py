"""Run the finite-prior, smoothed-prior, and boosted-tree research comparison."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from mtg_opponent_model.boosted_hands import BoostedHandModel
from mtg_opponent_model.hand_features import feature_matrix, HISTORY_FEATURES, BOARD_FEATURES
from mtg_opponent_model.hands import IncompatibleHistoryError
from mtg_opponent_model.priors import IndexedDeckPrior, SmoothedDeckPrior, load_card_info, BASIC_LANDS
from mtg_opponent_model.probability_evaluation import calibration_bins, probability_metrics
from mtg_opponent_model.replays import read_replay_rows, load_card_names, reconstruct_snapshots
from mtg_opponent_model.research_data import split_events, make_cases
from mtg_opponent_model.research_data import ResearchCase
from mtg_opponent_model.seventeenlands import SeventeenLandsLoader


def sample(cases: list[ResearchCase], count: int, seed: int) -> list[ResearchCase]:
    cases = sorted(cases, key=lambda case: case.game_id)
    random.Random(seed).shuffle(cases)
    return cases[:count]


def average_metrics(cases: list[ResearchCase], predictions: list[dict[str, float] | None],
                    cards: tuple[str, ...], indices: list[int]) -> dict:
    rows = [probability_metrics({c:p for c,p in predictions[i].items() if c in cards},
                                {c:n for c,n in cases[i].actual_hand.items() if c in cards}, cards)
            for i in indices if predictions[i] is not None]
    if not rows:
        return {'cases':0}
    return {'cases':len(rows), **{k:sum(row[k] for row in rows)/len(rows) for k in rows[0]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-snapshots',type=int,default=1500)
    parser.add_argument('--validation-snapshots',type=int,default=400)
    parser.add_argument('--test-snapshots',type=int,default=800)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('reports/blb_model_comparison.json'))
    args = parser.parse_args()
    if min(args.train_snapshots,args.validation_snapshots,args.test_snapshots,args.iterations)<1:
        parser.error('Snapshot limits and iterations must be positive')
    started=time.monotonic()
    paths=[Path('data/replay_sample_5000.BLB.PremierDraft.csv.gz'),Path('data/cards.csv'),
           Path('data/game_data_public.BLB.PremierDraft.csv.gz')]
    rows=read_replay_rows(paths[0],5000)
    names=load_card_names(paths[1]);metadata=load_card_info(paths[1])
    partition=split_events((row['draft_id'] for row in rows),args.seed)
    assert all(not partition[a]&partition[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    decks=list(SeventeenLandsLoader(paths[2],'BLB','PremierDraft',20000).load())
    training_decks=[d for d in decks if d.group_id not in partition['test']|partition['validation']]
    cards=tuple(sorted(c[5:] for c in rows[0] if c.startswith('deck_')))
    finite=IndexedDeckPrior(training_decks,cards)
    flexible=SmoothedDeckPrior(training_decks,cards,metadata)
    cases,case_stats=make_cases(rows,names,args.seed)
    selected={name:sample([c for c in cases if c.group_id in groups],limit,args.seed+offset)
              for offset,(name,groups,limit) in enumerate([
                  ('train',partition['train'],args.train_snapshots),
                  ('validation',partition['validation'],args.validation_snapshots),
                  ('test',partition['test'],args.test_snapshots)])}
    print('Selected cases:',{k:len(v) for k,v in selected.items()},flush=True)
    coverage=Counter();failure_reasons=Counter();coverage_examples=[];seen_games=set()
    for row in rows:
        if row['draft_id'] not in partition['test']:continue
        game=':'.join(row[k] for k in ('draft_id','match_number','game_number'))
        if game in seen_games:continue
        seen_games.add(game)
        snapshots,_=reconstruct_snapshots(row,names,30,allow_mulligans=True,allow_unknown_departures=True)
        recorded={k[5:]:int(v) for k,v in row.items() if k.startswith('deck_') and int(v)>0}
        for snapshot in snapshots:
            diagnosis=finite.diagnose(snapshot.public,recorded)
            coverage['audited_states']+=1
            coverage['finite_supported']+=diagnosis['compatible_decks']>0
            coverage['smoothed_supported']+=flexible.supports(snapshot.public)
            coverage['recorded_pool_violation_evaluation_only']+=diagnosis['recorded_pool_violation_evaluation_only']
            if not diagnosis['recorded_pool_violation_evaluation_only']:
                coverage['recorded_pool_consistent_states']+=1
                coverage['finite_supported_consistent']+=diagnosis['compatible_decks']>0
                coverage['smoothed_supported_consistent']+=flexible.supports(snapshot.public)
            failure_reasons[diagnosis['reason']]+=1
    print('Full held-out reconstruction/prior coverage:',dict(coverage),flush=True)
    # Cross-fit generative features for supervised training. An event's entire
    # fold is removed from the deck prior producing its training features.
    train_groups=sorted(partition['train'])
    folds=[set(train_groups[::2]),set(train_groups[1::2])]
    cross_priors=[]
    for excluded in folds:
        blocked=partition['validation']|partition['test']|excluded
        fold_decks=[d for d in decks if d.group_id not in blocked]
        assert not {d.group_id for d in fold_decks}&blocked
        cross_priors.append(SmoothedDeckPrior(fold_decks,cards,metadata))
    feature_sets={};labels={};retained={};prior_predictions={}
    for split,cases_in_split in selected.items():
        xs=[];ys=[];kept=[];predictions=[]
        for index,case in enumerate(cases_in_split):
            prior=(cross_priors[0] if case.group_id in folds[0] else cross_priors[1]) if split=='train' else flexible
            try:probs=prior.predict(case.history)
            except IncompatibleHistoryError:continue
            xs.append(feature_matrix(cards,metadata,case.history,probs,case.board))
            ys.append(np.array([case.actual_hand.get(card,0)>0 for card in cards],dtype=np.uint8))
            kept.append(case);predictions.append(probs)
            if (index+1)%250==0:print(f'{split}: {index+1} states featurized',flush=True)
        feature_sets[split]=np.concatenate(xs)
        labels[split]=np.concatenate(ys)
        retained[split]=kept;prior_predictions[split]=predictions
        print(split,'feature rows',feature_sets[split].shape,flush=True)
    model_artifacts={};test_predictions={'smoothed_prior':prior_predictions['test']}
    test_cases=retained['test'];finite_predictions=[]
    for case in test_cases:
        try:finite_predictions.append(finite.predict(case.history))
        except IncompatibleHistoryError:finite_predictions.append(None)
    test_predictions['historical_prior']=finite_predictions
    size_only={}
    test_predictions['hand_size_only']=[]
    from mtg_opponent_model.hands import PublicHistory
    for case in test_cases:
        size=case.history.steps[-1].hand_size
        if size not in size_only:
            size_only[size]=flexible.predict(PublicHistory(size,()))
        test_predictions['hand_size_only'].append(size_only[size])
    test_predictions['all_absent']=[dict.fromkeys(cards,0.0) for _ in test_cases]
    for include_board,method in [(False,'boosted_history'),(True,'boosted_board')]:
        width=len(HISTORY_FEATURES)+(len(BOARD_FEATURES) if include_board else 0)
        with threadpool_limits(limits=4):
            model=BoostedHandModel(args.seed,args.iterations).fit(feature_sets['train'][:,:width],labels['train'])
            model.calibrate(feature_sets['validation'][:,:width],labels['validation'])
            for calibrated in (False,True):
                values=model.predict(feature_sets['test'][:,:width],calibrated=calibrated).reshape(-1,len(cards))
                name=method+('_calibrated' if calibrated else '_raw')
                test_predictions[name]=[dict(zip(cards,row.tolist())) for row in values]
        model_artifacts[method]=model
        print('Trained and calibrated:',method,flush=True)
    all_indices=list(range(len(test_cases)))
    common=[i for i,p in enumerate(finite_predictions) if p is not None]
    rescued=[i for i,p in enumerate(finite_predictions) if p is None]
    clean=[i for i,c in enumerate(test_cases) if not finite.diagnose(c.history,c.recorded_deck)['recorded_pool_violation_evaluation_only']]
    no_basics=tuple(c for c in cards if c not in BASIC_LANDS)
    metrics={scope:{method:average_metrics(test_cases,preds,pool,indices) for method,preds in test_predictions.items()}
             for scope,pool,indices in [('all_audited_sample',cards,all_indices),('common_historical_support',cards,common),
                                        ('rescued_sample',cards,rescued),('recorded_pool_consistent_sample',cards,clean),
                                        ('excluding_basic_lands',no_basics,all_indices)]}
    calibration={method:calibration_bins([p for p in preds if p is not None],
                                        [c.actual_hand for c,p in zip(test_cases,preds) if p is not None],cards)
                 for method,preds in test_predictions.items()}
    # Paired event bootstrap differences, never treating candidate rows as
    # independent games. One sampled state per game gives each game equal weight.
    paired={}
    rng=np.random.default_rng(args.seed)
    groups=sorted({c.group_id for c in test_cases})
    for a,b in [('smoothed_prior','historical_prior'),('boosted_board_calibrated','smoothed_prior'),
                ('boosted_board_calibrated','boosted_history_calibrated')]:
        indices=[i for i in all_indices if test_predictions[a][i] is not None and test_predictions[b][i] is not None]
        differences={}
        for metric in ('Recall@20','Brier'):
            totals=Counter();counts=Counter()
            for i in indices:
                group=test_cases[i].group_id
                totals[group]+=(probability_metrics(test_predictions[a][i],test_cases[i].actual_hand,cards)[metric]
                                -probability_metrics(test_predictions[b][i],test_cases[i].actual_hand,cards)[metric])
                counts[group]+=1
            events=sorted(counts);draws=[]
            for _ in range(1000):
                chosen=rng.choice(events,size=len(events),replace=True)
                draws.append(sum(totals[e] for e in chosen)/sum(counts[e] for e in chosen))
            differences[metric]={'estimate':sum(totals.values())/sum(counts.values()),
                                 'ci95':np.quantile(draws,[.025,.975]).tolist(),'cases':len(indices)}
        paired[f'{a} minus {b}']=differences
    example=[]
    for case_index in rescued[:3]:
        case=test_cases[case_index]
        example.append({'game':case.game_id,'turn':case.history.steps[-1].turn,
                        'public_input':{'history':asdict(case.history),'board':case.board},
                        'actual_hand_evaluation_only':dict(case.actual_hand),
                        'top_smoothed':sorted(test_predictions['smoothed_prior'][case_index].items(),key=lambda x:(-x[1],x[0]))[:10],
                        'top_boosted':sorted(test_predictions['boosted_board_calibrated'][case_index].items(),key=lambda x:(-x[1],x[0]))[:10]})
    artifacts=Path('data/hand_research_models.joblib')
    joblib.dump({'models':model_artifacts,'prior':flexible,'cards':cards,'metadata':metadata,
                 'feature_names':HISTORY_FEATURES+BOARD_FEATURES,'sklearn_version':sklearn.__version__},artifacts)
    artifact_size=artifacts.stat().st_size
    reloaded=joblib.load(artifacts)
    with threadpool_limits(limits=4):
        restored=reloaded['models']['boosted_board'].predict(feature_sets['test'][:len(cards)])
    assert np.allclose(restored,[test_predictions['boosted_board_calibrated'][0][card] for card in cards],atol=1e-10)
    if example:
        Path('data/public_hand_example.json').write_text(json.dumps(example[0]['public_input'],indent=2)+'\n')
    split_path=Path('data/hand_research_split.json')
    split_path.write_text(json.dumps({k:sorted(v) for k,v in partition.items()},indent=2)+'\n')
    report={'source':'17Lands public datasets, CC BY 4.0','settings':vars(args)|{'output':str(args.output)},
            'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'split':{'events':{k:len(v) for k,v in partition.items()},
                     'fingerprints':{k:hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest() for k,v in partition.items()},
                     'event_overlap':0,'prior_training_decks':len(training_decks),
                     'selected_games':{k:len(v) for k,v in selected.items()},'retained_cases':{k:len(v) for k,v in retained.items()},
                     'sampling':'one seeded random audited state per game, then seeded bounded game samples',
                     'cross_fitting':'two event folds for training generative features; validation/test excluded from every prior'},
            'reconstruction':case_stats,'full_test_coverage':dict(coverage),'full_test_failure_reasons':dict(failure_reasons),
            'sample_support':{'total':len(test_cases),'historical':len(common),'rescued':len(rescued),'recorded_pool_consistent':len(clean)},
            'metrics':metrics,'calibration':calibration,'paired_event_bootstrap':paired,
            'feature_names':list(HISTORY_FEATURES+BOARD_FEATURES),'examples':example,
            'generative_parameters':{'concentration':flexible.concentration,'shrinkage':flexible.shrinkage,
                                     'floor':flexible.floor,'color_groups':len(flexible.alpha),
                                     'size_land_components':len(flexible.components)},
            'tree_parameters':{method:model.estimator.get_params() for method,model in model_artifacts.items()},
            'calibration_parameters':{method:{'slope':float(model.calibrator.coef_[0,0]),
                                               'intercept':float(model.calibrator.intercept_[0])}
                                      for method,model in model_artifacts.items()},
            'artifact_roundtrip_verified':True,
            'artifacts':{'path':str(artifacts),'bytes':artifact_size,
                         'sha256':hashlib.sha256(artifacts.read_bytes()).hexdigest()},
            'elapsed_seconds':time.monotonic()-started,
            'limitations':['File-prefix BLB sample; no cross-set generalization claim',
                           'Fixed hyperparameters, no test-based selection; sigmoid calibration uses validation labels only',
                           'All 276 candidate cards retained; no negative sampling or class reweighting',
                           'Recorded deck diagnostics never affect model inputs or main sample selection',
                           'No hand-size/copy joint consistency guarantee for independent tree scores; finite prior zero-probability constraints are enforced',
                           'Flexible prior keeps empirical size/land shapes; it does not handle returns, tutors, exile casting, or generated cards',
                           'Groups are color pairs, not learned tribal archetypes; no card rules-text features',
                           'Coverage is reported on all audited held-out states, accuracy on a bounded one-state-per-game sample',
                           'Mulligan kept hands and unidentified departures retain the baseline approximations',
                           'No website or previous benchmark results are replaced by this experiment']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Test results:',json.dumps(metrics['all_audited_sample'],indent=2),flush=True)
    print('Saved report:',args.output,'models:',artifacts,flush=True)


if __name__=='__main__':
    main()
