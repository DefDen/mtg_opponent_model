import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('sklearn')

from mtg_opponent_model.hands import PublicHistory, PublicStep
from mtg_opponent_model.hand_features import feature_matrix,HISTORY_FEATURES,BOARD_FEATURES
from mtg_opponent_model.priors import CardInfo
from mtg_opponent_model.probability_evaluation import probability_metrics,calibration_bins
from mtg_opponent_model.research_data import split_events
from mtg_opponent_model.boosted_hands import BoostedHandModel


def test_public_feature_dimensions_and_board_ablation():
    cards=['A','B'];metadata={'A':CardInfo('R','Creature',2),'B':CardInfo('G','Land',0)}
    history=PublicHistory(2,(PublicStep(1,'target',0,('A',),1),))
    board={p:{'life':20,'lands':['B'],'creatures':[],'other':[]} for p in ('target','observer')}
    both=feature_matrix(cards,metadata,history,{'A':.2,'B':.5},board)
    no_board=feature_matrix(cards,metadata,history,{'A':.2,'B':.5},{},include_board=False)
    assert both.shape==(2,len(HISTORY_FEATURES)+len(BOARD_FEATURES))
    assert no_board.shape==(2,len(HISTORY_FEATURES))
    assert np.array_equal(both[:,:len(HISTORY_FEATURES)],no_board)
    assert both[0,HISTORY_FEATURES.index('observed_copies')]==1
    assert both[1,len(HISTORY_FEATURES)+BOARD_FEATURES.index('candidate_on_target_board')]==1


def test_event_split_disjoint_and_deterministic():
    split=split_events([str(i) for i in range(30)])
    assert split==split_events(reversed([str(i) for i in range(30)]))
    assert not split['train']&split['test']
    assert not split['train']&split['validation']
    assert not split['test']&split['validation']
    assert len(set.union(*split.values()))==30


def test_probability_calibration_metrics_have_known_values():
    result=probability_metrics({'A':.75,'B':.25},{'A':1},['A','B'])
    assert result['Brier']==pytest.approx(.0625)
    assert result['LogLoss']==pytest.approx(-np.log(.75))
    bins=calibration_bins([{'A':.75,'B':.25}],[{'A':1}],['A','B'])
    assert bins['expected_calibration_error']==pytest.approx(.25)
    assert bins['card_snapshot_pairs']==2


def test_boosting_and_calibration_preserve_hard_prior_zeros():
    rng=np.random.default_rng(4)
    x=rng.random((240,3),dtype=np.float32)
    y=(x[:,0]>.5).astype(int)
    model=BoostedHandModel(iterations=3).fit(x[:160],y[:160])
    model.calibrate(x[160:],y[160:])
    test=x[:8].copy();test[0,0]=0
    for calibrated in (False,True):
        probabilities=model.predict(test,calibrated)
        assert probabilities[0]==0
        assert np.all((probabilities>=0)&(probabilities<=1))
