import copy

import numpy as np
import pytest

from self_assembling_games import model, System
from self_assembling_games.core import Outcome


def test_release_preserves_learned_side_and_rng():
    source = model('cue.demo').instantiate(seed=3).advance(15)
    before = source.snapshot()
    child = source.transition(model('sig.demo'), {'kind': 'release_fixed', 'fixed_weight': 9, 'other_weight': 1})
    assert source.snapshot() == before
    assert child.position == source.position
    old, new = source._backend.simulation.runtime, child._backend.simulation.runtime
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    for identity, table in old.tables.items():
        if table.module == 'receiver':
            assert table.to_dict() == new.tables[identity].to_dict()
        else:
            assert new.tables[identity].current is None
            assert sorted(new.tables[identity].weights) == [1, 9]
    restored = System.restore(child.snapshot())
    child.advance(8)
    restored.advance(8)
    assert child.snapshot() == restored.snapshot()


def test_fork_freeze_and_thaw_preserve_policy():
    source = model('sig.demo').instantiate(seed=1).advance(10)
    fork = source.fork()
    state = copy.deepcopy(fork._backend.simulation.runtime.tables['sender/0'].weights)
    fork.set_trainable(['sender'], False).advance(20)
    assert np.array_equal(state, fork._backend.simulation.runtime.tables['sender/0'].weights)
    fork.set_trainable(['sender'], True)
    assert source.position == 10
    assert System.restore(fork.snapshot()).snapshot() == fork.snapshot()
    with pytest.raises(ValueError):
        fork.set_trainable(['unknown'], True)


def test_reject_incompatible_carry():
    with pytest.raises(ValueError, match='compatible organisation'):
        model('sig.demo').instantiate().transition(model('nand.demo'), {'kind':'identity'})


def test_owner_returns_are_complete_and_separate():
    outcome = Outcome(True, 7, owner_returns={'a': -1, 'b': 2})
    assert outcome.payoff == 7
    assert outcome.returns_for(['a', 'b', 'b']) == {'a': -1, 'b': 2}
    with pytest.raises(ValueError):
        outcome.returns_for(['a'])


def test_identity_carry_retains_imported_state_without_original_artifact(tmp_path):
    from self_assembling_games.component_artifacts import export_component

    path = export_component(model('nand.demo').instantiate(seed=11).advance(5), tmp_path)
    source = model('tr-log.demo', interaction={'parameters':{
        'template_mode':'artifact', 'templates':{'template':str(path)},
    }}).instantiate(seed=3).advance(9)
    before = source.snapshot()
    target = model(source.model, execution={'budget':17}, source={
        'assumptions':source.model.to_dict()['source']['assumptions'] + ['Identity continuation control.']})
    path.unlink()
    child = source.transition(target, {'kind':'identity'})
    assert source.snapshot() == before
    assert child.position == 9
    assert child.model.to_dict() == target.to_dict()
    assert child.snapshot()['state']['profile']['assumptions'] == target.to_dict()['source']['assumptions']
    assert child.snapshot()['state']['profile']['plays'] == 17
    restored = System.restore(child.snapshot())
    child.advance(8)
    restored.advance(8)
    source.advance(8)
    assert restored.snapshot() == child.snapshot()
    assert child.snapshot()['state']['runtime'] == source.snapshot()['state']['runtime']
