"""Explicit transformations of acquired conditional state at encounter boundaries."""
from __future__ import annotations

import copy
import math

from .core import Runtime
from .domain import Model
from .metrics import Metrics

_FIXED_SHAPE = {
    'SIG', 'CUE', 'MAN', 'SCOD', 'COD', 'NAND', 'HIER', 'LC-LOG',
    'GEN-APPROX', 'TR-LOG', 'COMP-NAND', 'TR-ORD', 'TR-PART', 'ADD-COMP',
}


def _conditional(system):
    from .simulation import Simulation
    simulation = getattr(system._backend, 'simulation', None)
    if not isinstance(simulation, Simulation):
        raise ValueError('This transition requires a conditional-table provider')
    return simulation


def set_trainable(system, modules, enabled):
    simulation = _conditional(system)
    if simulation.profile.model not in _FIXED_SHAPE:
        raise ValueError('Trainability transitions require a supported fixed-shape model')
    simulation.runtime.set_trainable(list(modules), enabled)
    simulation.history.append({'intervention': 'trainability', 'modules': list(modules),
                               'enabled': enabled, 'position': system.position})


def transition_system(system, target_model, transition):
    """Create a valid descendant; no source state is mutated."""
    from .system import System
    if not isinstance(transition, dict):
        raise ValueError('A transition requires a named declaration')
    kind = transition.get('kind')
    if kind not in {'identity', 'release_fixed'}:
        raise ValueError('Unsupported state transition')
    allowed = {'kind'} if kind == 'identity' else {'kind', 'fixed_weight', 'other_weight'}
    if transition.keys() - allowed:
        raise ValueError('Unknown transition parameters')
    source = _conditional(system)
    target = target_model if isinstance(target_model, Model) else Model.from_dict(target_model)
    before, after = system.model.to_dict(), target.to_dict()
    for field in ('organisation', 'realisation', 'adaptation', 'resources'):
        if before[field] != after[field]:
            raise ValueError('State carry requires compatible ' + field)
    if before['execution']['parameters'] != after['execution']['parameters']:
        raise ValueError('State carry preserves the execution schedule')
    if kind == 'identity':
        if before['interaction'] != after['interaction']:
            raise ValueError('Identity transition requires the same interaction')
    else:
        old, new = before['interaction'], after['interaction']
        if old['game'] not in {'CUE', 'MAN'} or new['game'] != 'SIG':
            raise ValueError('release_fixed supports CUE/MAN to SIG')
        if source.profile.learner != {'rule': 'R1'} and (
            source.profile.learner.get('rule') != 'R1'
            or source.profile.learner.get('error', 0) != 0
        ):
            raise ValueError('release_fixed requires proportional R1 without exploration error')
        old_parameters = {k: v for k, v in old['parameters'].items() if k != 'fixed_map'}
        if old_parameters != new['parameters']:
            raise ValueError('release_fixed requires identical observation/action/payoff domains')
        for key in ('fixed_weight', 'other_weight'):
            value = transition.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('release_fixed requires explicit positive finite weights')
    parent = system.snapshot()
    child = System(target, seed=system.seed, protocol=system.protocol)
    destination = _conditional(child)
    destination.runtime = Runtime.restore(source.runtime.state())
    destination.world = copy.deepcopy(source.world)
    destination.interaction = copy.deepcopy(source.interaction)
    destination.lineage = copy.deepcopy(source.lineage)
    destination.history = copy.deepcopy(source.history)
    destination.metrics = Metrics.restore(source.metrics.state())
    destination.status = source.status
    if kind == 'release_fixed':
        module = 'sender' if source.profile.model == 'CUE' else 'receiver'
        for table in destination.runtime.tables.values():
            if table.module == module:
                if table.current is None:
                    raise ValueError('Release source is missing a fixed assignment')
                chosen = table.index(table.current)
                table.weights[:] = transition['other_weight']
                table.weights[chosen] = transition['fixed_weight']
                table.current = None
                table.validate()
        destination.runtime.set_trainable([module], True)
    child.lineage = {'parent_checkpoint_hash': parent['checkpoint_hash'],
                     'operation': 'transition', 'transition': copy.deepcopy(transition)}
    destination.history.append({**child.lineage, 'position': child.position})
    # Exercise the normal provenance and model/state compatibility gates now.
    return System.restore(child.snapshot())
