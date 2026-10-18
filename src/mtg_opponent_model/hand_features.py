"""Public-only, candidate-card features for a shared supervised hand model."""
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .hands import PublicHistory
from .priors import BASIC_LANDS, CardInfo

HISTORY_FEATURES = ('prior_probability','observed_copies','turns_since_reveal','hand_size',
                    'opening_size','turn','target_turn','draws','named_departures','unknown_departures',
                    'mana_value','is_basic','is_land','is_creature','is_instant','is_sorcery',
                    'is_artifact','is_enchantment','is_planeswalker','color_W','color_U','color_B','color_R','color_G')
BOARD_FEATURES = ('target_life','observer_life','target_land_count','observer_land_count',
                  'target_creature_count','observer_creature_count','target_other_count','observer_other_count',
                  'candidate_on_target_board','candidate_on_observer_board',
                  'target_land_W','target_land_U','target_land_B','target_land_R','target_land_G')


def feature_matrix(cards: Sequence[str], metadata: Mapping[str, CardInfo], history: PublicHistory,
                   prior: Mapping[str, float], board: Mapping[str, dict], *,
                   include_board: bool = True) -> np.ndarray:
    """No labels or recorded deck input. Candidate names serve metadata lookup only."""
    seen = Counter(c for step in history.steps for c in step.departed)
    last_seen = {card: step.turn for step in history.steps for card in step.departed}
    step = history.steps[-1]
    shared = [step.hand_size, history.opening_hand_size, step.turn, int(step.active_player == 'target'),
              sum(s.draws for s in history.steps), sum(len(s.departed) for s in history.steps),
              sum(s.unknown_departures for s in history.steps)]
    board_shared = []
    target_cards = observer_cards = Counter()
    if include_board:
        target, observer = board['target'], board['observer']
        board_shared = [target['life'], observer['life'], len(target['lands']),len(observer['lands']),
                        len(target['creatures']),len(observer['creatures']),len(target['other']),len(observer['other'])]
        target_cards = Counter(c for zone in ('lands','creatures','other') for c in target[zone])
        observer_cards = Counter(c for zone in ('lands','creatures','other') for c in observer[zone])
        land_colors = [sum(color in metadata.get(c, CardInfo()).colors for c in target['lands']) for color in 'WUBRG']
    rows = []
    for card in cards:
        info = metadata.get(card, CardInfo())
        row = [prior.get(card, 0), seen[card], step.turn-last_seen.get(card, 0), *shared,
               info.mana_value, int(card in BASIC_LANDS),
               *[int(kind in info.types) for kind in ('Land','Creature','Instant','Sorcery','Artifact','Enchantment','Planeswalker')],
               *[int(color in info.colors) for color in 'WUBRG']]
        if include_board:
            row += [*board_shared, target_cards[card], observer_cards[card], *land_colors]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)
