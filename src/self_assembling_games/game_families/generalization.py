"""O'Connor's approximation problem with explicitly spread reinforcement."""
from __future__ import annotations

import math

from ..core import Outcome
from ..generalization import reinforce_contexts
from .common import rows


def resolve_parameters(model, parameters):
    parameters.setdefault('coordinates', [0, 1])
    parameters.setdefault('state_probabilities', [1 / len(parameters['coordinates'])] * len(parameters['coordinates']))
    parameters.setdefault('payoff_kind', 'binary')
    if parameters['payoff_kind'] == 'gaussian':
        parameters.setdefault('payoff_height', 1.0)
        parameters.setdefault('payoff_width', 2.0)
    return parameters


def initialize(instance):
    p = instance.parameters
    rows(instance.runtime, 'actor', list(range(len(p['coordinates']))),
         list(range(len(p['coordinates']))), instance.learner)


def encounter(instance):
    rt, p = instance.runtime, instance.parameters
    s = rt.chance('nature', p['state_probabilities'])
    d = rt.decide('actor', s, s)
    distance = abs(p['coordinates'][s] - p['coordinates'][d.option])
    reward = (2.0 if s == d.option else 1.0) if p['payoff_kind'] == 'binary' else (
        p['payoff_height'] * math.exp(-0.5 * (distance / p['payoff_width']) ** 2))
    outcome = Outcome(s == d.option, reward)
    reinforce_contexts(rt, d, outcome, p['coordinates'], instance.learner['generalization'])
    return outcome


def validate(profile):
    p = resolve_parameters(profile.model, dict(profile.parameters))
    allowed_parameters = {
        'coordinates', 'state_probabilities', 'payoff_kind', 'window', 'owners',
    }
    if p.get('payoff_kind') == 'gaussian':
        allowed_parameters.update({'payoff_height', 'payoff_width'})
    if p.keys() - allowed_parameters:
        raise ValueError('Unknown approximation parameters: ' + ', '.join(sorted(p.keys() - allowed_parameters)))
    if profile.learner.keys() - {'rule', 'error', 'generalization'}:
        raise ValueError('Unknown approximation learning parameters')
    coordinates = p['coordinates']
    if not isinstance(coordinates, list) or not coordinates or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in coordinates
    ) or len(set(coordinates)) != len(coordinates):
        raise ValueError('Coordinates must be distinct finite numbers')
    from ..registry import validate_distribution
    validate_distribution(p['state_probabilities'], len(coordinates))
    if p['payoff_kind'] not in {'binary', 'gaussian'}:
        raise ValueError('Unknown approximation payoff')
    if p['payoff_kind'] == 'binary' and len(coordinates) != 2:
        raise ValueError('Binary approximation has two states and actions')
    if p['payoff_kind'] == 'gaussian':
        for key in ('payoff_height', 'payoff_width'):
            if isinstance(p[key], bool) or not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or p[key] <= 0:
                raise ValueError('Gaussian payoff height and width must be positive finite values')
    g = profile.learner.get('generalization', {})
    if g.get('kind') == 'binary':
        if len(coordinates) != 2 or set(g) != {'kind', 'alpha'} or isinstance(g.get('alpha'), bool) or not isinstance(g.get('alpha'), (int, float)) or not 0 <= g['alpha'] <= 1:
            raise ValueError('Binary generalization requires alpha in [0,1]')
    elif g.get('kind') == 'gaussian':
        if set(g) != {'kind', 'width'} or isinstance(g.get('width'), bool) or not isinstance(g.get('width'), (int, float)) or not math.isfinite(g['width']) or g['width'] < 0:
            raise ValueError('Gaussian generalization requires a finite nonnegative width')
    else:
        raise ValueError('Generalization requires a binary or Gaussian kernel')
    if profile.learner.get('rule') != 'R1' or profile.learner.get('error', 0) != 0:
        raise ValueError('Approximation profiles use proportional R1 without exploration error')
