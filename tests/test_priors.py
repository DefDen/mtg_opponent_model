import pytest

np = pytest.importorskip('numpy')
from mtg_opponent_model.data import DeckRecord
from mtg_opponent_model.hands import HandModel, PublicHistory, PublicStep, IncompatibleHistoryError
from mtg_opponent_model.priors import CardInfo, IndexedDeckPrior, SmoothedDeckPrior


META = {'L': CardInfo('R', 'Land'), 'A': CardInfo('R', 'Creature'),
        'B': CardInfo('R', 'Instant'), 'Novel': CardInfo('R', 'Creature')}


def test_indexed_prior_matches_original_including_unidentified_losses():
    decks = [DeckRecord('a','a',{'A':2,'B':2}),DeckRecord('b','b',{'A':1,'L':3})]
    original = HandModel(decks)
    indexed = IndexedDeckPrior(decks, ['L','B','A'])
    histories = [PublicHistory(2, ()),
                 PublicHistory(2, (PublicStep(1,'target',0,('A',),1),)),
                 PublicHistory(3, (PublicStep(1,'observer',0,(),2,1),
                                   PublicStep(1,'target',0,('A',),1)))]
    for history in histories:
        assert indexed.predict(history) == pytest.approx(original.predict(history))


def test_smoothed_prior_exact_land_and_dirichlet_marginals():
    decks = [DeckRecord('d','g',{'L':1,'A':1,'B':1})]
    model = SmoothedDeckPrior(decks, ['L','A','B'], META, concentration=50)
    assert model.predict(PublicHistory(1,())) == pytest.approx({'L':1/3,'A':1/3,'B':1/3})
    full = model.predict(PublicHistory(3,()))
    assert full['L'] == pytest.approx(1)
    assert full['A'] == pytest.approx(1-25*26/(50*51))
    played = PublicHistory(3,(PublicStep(1,'target',0,('A',),2),))
    assert model.predict(played) == pytest.approx({'L':1,'A':26/51,'B':25/51})


def test_new_combinations_supported_without_inventing_slots():
    decks = [DeckRecord('a','a',{'L':1,'A':2}),DeckRecord('b','b',{'L':1,'B':2})]
    history = PublicHistory(3,(PublicStep(1,'target',0,('A','B'),1),))
    with pytest.raises(IncompatibleHistoryError):
        IndexedDeckPrior(decks,META).predict(history)
    smooth = SmoothedDeckPrior(decks,META,META)
    result = smooth.predict(history)
    assert result['L'] == pytest.approx(1)
    assert result['A'] == result['B'] == result['Novel'] == 0
    impossible = PublicHistory(3,(PublicStep(1,'target',0,('A','A','A'),0),))
    with pytest.raises(IncompatibleHistoryError):
        smooth.predict(impossible)


def test_pool_floor_handles_unseen_legal_cards_but_not_out_of_pool_cards():
    model = SmoothedDeckPrior([DeckRecord('d','g',{'L':1,'A':2})],META,META)
    assert model.predict(PublicHistory(1,()))['Novel'] > 0
    assert model.predict(PublicHistory(3,(PublicStep(1,'target',0,('Novel',),2),)))['L'] > 0
    with pytest.raises(IncompatibleHistoryError):
        model.predict(PublicHistory(3,(PublicStep(1,'target',0,('Unknown',),2),)))


def test_private_recorded_deck_only_changes_diagnosis_not_predictions():
    prior = IndexedDeckPrior([DeckRecord('a','a',{'A':2,'L':1}),
                              DeckRecord('b','b',{'B':2,'L':1})],META)
    history = PublicHistory(3,(PublicStep(1,'target',0,('A','B'),1),))
    valid = prior.diagnose(history,{'A':1,'B':1,'L':1})
    reused = prior.diagnose(history,{'A':2,'L':1})
    assert valid['reason'] == 'unseen_card_combination'
    assert reused['reason'] == 'exceeds_recorded_single_deck_accounting'
    with pytest.raises(IncompatibleHistoryError):
        prior.predict(history)


def test_smoothing_still_rejects_library_exhaustion():
    model = SmoothedDeckPrior([DeckRecord('d','g',{'A':2})],['A'],META)
    history = PublicHistory(2,(PublicStep(1,'observer',0,(),1,1),
                               PublicStep(1,'target',1,(),2)))
    with pytest.raises(IncompatibleHistoryError):
        model.predict(history)
