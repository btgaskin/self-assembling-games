import json

import pytest

from self_assembling_games.domain import Model, experiment, model, resolve
from self_assembling_games.system import System


def test_declarations_are_detached_and_do_not_build(monkeypatch):
    from self_assembling_games import builtin

    monkeypatch.setattr(
        builtin,
        "create_backend",
        lambda *a: pytest.fail("construction during resolution"),
    )
    m = model("sig.demo")
    d = resolve(m)
    d["interaction"]["parameters"]["states"] = 999
    assert resolve(m) != d
    assert json.loads(json.dumps(resolve(m))) == resolve(m)
    assert experiment(m, seeds=[2, 5], budget=3).to_dict()["seeds"] == [2, 5]


def test_system_exact_continuation_and_evaluation_isolation():
    first = model("sig.demo").instantiate(seed=23)
    first.advance(9)
    state = first.snapshot()
    assert first.evaluate()["protocol"] == "exact"
    assert first.snapshot() == state
    resumed = System.restore(json.loads(json.dumps(state)))
    first.advance(11)
    resumed.advance(11)
    assert resumed.snapshot() == first.snapshot()


def test_checkpoint_integrity_and_provider_identity():
    state = model("sig.demo").instantiate().snapshot()
    state["seed"] = 99
    with pytest.raises(ValueError, match="integrity"):
        System.restore(state)


@pytest.mark.parametrize("budget", [-1, float("inf"), True, 1.2])
def test_discrete_budget_validation(budget):
    data = model("sig.demo").to_dict()
    data["execution"]["budget"] = budget
    with pytest.raises(ValueError):
        Model.from_dict(data)


def test_invalid_replicate_seeds():
    with pytest.raises(ValueError, match="Seeds"):
        experiment("sig.demo", seeds=[1, 1])


def test_provenance_is_owned_by_each_runtime():
    from self_assembling_games.simulation import provenance
    metadata = provenance()
    metadata["scipy"] = "solver-specific"
    assert "scipy" not in provenance()
