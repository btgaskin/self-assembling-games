import numpy as np
import pytest

from self_assembling_games import model, System
from self_assembling_games.core import Outcome
from self_assembling_games.generalization import reinforce_contexts


def test_exact_spread_and_frozen_destination():
    system = model('oconnor.generalization.binary').instantiate()
    rt = system._backend.simulation.runtime
    decision = rt.decide('actor', 0, 0)
    reinforce_contexts(rt, decision, Outcome(True, 2), [0, 1], {'kind':'binary','alpha':.1})
    assert rt.tables['actor/0'].weights[decision.option] == 3
    assert rt.tables['actor/1'].weights[decision.option] == 1.2
    rt.tables['actor/1'].frozen = True
    before = rt.tables['actor/1'].weights.copy()
    reinforce_contexts(rt, decision, Outcome(True, 2), [0,1], {'kind':'binary','alpha':.1})
    assert np.array_equal(before, rt.tables['actor/1'].weights)


@pytest.mark.parametrize('alpha', [0, 1])
def test_binary_invariants_and_restore(alpha):
    system = model('oconnor.generalization.binary', adaptation={'parameters':{'generalization':{'alpha':alpha}}}).instantiate(seed=3)
    system.advance(40)
    if alpha == 1:
        rt = system._backend.simulation.runtime
        assert np.array_equal(rt.tables['actor/0'].weights, rt.tables['actor/1'].weights)
        assert system.evaluate()['expected_net_payoff'] == pytest.approx(1.5)
        assert system.evaluate()['normalized_expected_payoff'] == pytest.approx(.75)
    restored = System.restore(system.snapshot())
    system.advance(10)
    restored.advance(10)
    assert system.snapshot() == restored.snapshot()


def test_gaussian_not_normalized_or_wrapped():
    system = model('oconnor.generalization.gaussian').instantiate()
    rt = system._backend.simulation.runtime
    d = rt.decide('actor', 0, 0)
    reinforce_contexts(rt, d, Outcome(True, 1), list(range(20)), {'kind':'gaussian','width':2})
    assert rt.tables['actor/0'].weights[d.option] == 2
    assert rt.tables['actor/1'].weights[d.option] == pytest.approx(1 + np.exp(-1/8))
    assert rt.tables['actor/19'].weights[d.option] == pytest.approx(1)


def test_zero_width_only_current_context():
    system = model('oconnor.generalization.gaussian', adaptation={'parameters':{'generalization':{'width':0}}}).instantiate()
    rt = system._backend.simulation.runtime
    d = rt.decide('actor', 0, 0)
    reinforce_contexts(rt,d,Outcome(True,1),list(range(20)),{'kind':'gaussian','width':0})
    assert rt.tables['actor/0'].total == 21
    assert rt.tables['actor/1'].total == 20
