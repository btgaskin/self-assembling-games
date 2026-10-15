"""The built-in provider binds resolved declarations to existing kernels."""

import copy

import numpy as np
import pytest

from self_assembling_games.builtin import create_backend, resolve_model, restore_backend
from self_assembling_games.finite import FiniteGame
from self_assembling_games.finite_profiles import adaptive_profiles
from self_assembling_games.registry import profiles
from self_assembling_games.simulation import build


@pytest.mark.parametrize("preset", [p["id"] for p in profiles() + adaptive_profiles()])
def test_every_preset_resolves_without_finite_enumeration(preset, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Resolution attempted exact finite enumeration")

    monkeypatch.setattr(FiniteGame, "_enumerate_plans", forbidden)
    monkeypatch.setattr(FiniteGame, "compile", forbidden)
    model = resolve_model(preset)
    assert model["id"] == preset
    assert model["runtime"] == "builtin"
    assert resolve_model(model) == model
    assert "profile" not in model


@pytest.mark.parametrize(
    "preset",
    [
        "sig.demo",
        "band.arp.demo",
        "sk-sig.r1.demo",
        "sk-sig.strategy-r1.demo",
        "sk-sig.evolution.demo",
        "sk-sig.shared.demo",
        "sk-comp.grouped.demo",
        "sk-chain.translator.demo",
        "sk-sig.mutation.demo",
        "sk-sig.cost-listening.prose_derived.evolution.demo",
        "sk-sig.cost-listening.table_literal.strategy-r1.demo",
        "ml-coord.deterministic.demo",
        "ml-coord.wf.demo",
        "barrett.sig.evolution.demo",
    ],
)
def test_numerical_mechanisms_match_existing_kernels(preset):
    backend = create_backend(resolve_model(preset), 17, {})
    legacy = build(preset, 17)
    amount = 0.125 if backend.clock == "time" else 4
    backend.advance(amount)
    legacy.run(
        **{
            {"encounters": "plays", "generations": "generations", "time": "duration"}[
                backend.clock
            ]: amount
        }
    )
    if backend.clock == "time":
        for name in legacy.frequencies:
            assert np.array_equal(
                legacy.frequencies[name], backend.simulation.frequencies[name]
            )
        assert legacy.evaluate() == backend.evaluate()
    elif backend.clock == "generations":
        assert np.array_equal(
            legacy._realisation.frequencies, backend.simulation._realisation.frequencies
        )
        assert legacy.evaluate() == backend.evaluate()
    elif hasattr(legacy, "weights"):
        for name in legacy.weights:
            assert np.array_equal(
                legacy.weights[name], backend.simulation.weights[name]
            )
        assert legacy.evaluate() == backend.evaluate()
    else:
        old_state, new_state = (
            legacy.runtime.state(),
            backend.simulation.runtime.state(),
        )
        from self_assembling_games.learners import resolved_parameters

        for state in (old_state, new_state):
            for table in state["tables"].values():
                table["learner"] = resolved_parameters(table["learner"])
        assert old_state == new_state
        assert legacy.metrics.state() == backend.simulation.metrics.state()


@pytest.mark.parametrize(
    "preset",
    ["sig.demo", "sk-sig.r1.demo", "sk-sig.evolution.demo", "ml-coord.wf.demo"],
)
def test_backend_checkpoint_continues_numerical_state(preset):
    model = resolve_model(preset)
    backend = create_backend(model, 29, {})
    amount = 0.125 if backend.clock == "time" else 3
    backend.advance(amount)
    checkpoint = copy.deepcopy(backend.snapshot())
    restored = restore_backend(model, checkpoint)
    assert restored.snapshot() == checkpoint
    backend.advance(amount)
    restored.advance(amount)
    assert restored.snapshot() == backend.snapshot()


def test_large_declaration_retains_sampled_capability_without_exact_arrays(monkeypatch):
    model = resolve_model(
        "sk-sig.r1.demo",
        {
            "interaction": {
                "parameters": {
                    "states": 8,
                    "signals": 8,
                    "actions": 8,
                    "state_probabilities": [0.125] * 8,
                }
            }
        },
    )
    assert "evaluate.sampled" in model["capabilities"]
    assert "evaluate.exact" not in model["capabilities"]
    original = FiniteGame._enumerate_plans

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampled lifecycle attempted exact strategy enumeration")

    monkeypatch.setattr(FiniteGame, "_enumerate_plans", forbidden)
    backend = create_backend(model, 0, {})
    backend.advance(3)
    assert backend.evaluate(protocol="sampled", samples=5)["samples"] == 5
    checkpoint = backend.snapshot()
    restored = restore_backend(model, checkpoint)
    assert restored.snapshot() == checkpoint
    restored.inspect()
    monkeypatch.setattr(FiniteGame, "_enumerate_plans", original)
    with pytest.raises(ValueError, match="exact finite capability"):
        backend.evaluate(protocol="exact")


def test_haplotype_expression_keeps_joint_individual_binding():
    model = resolve_model("ml-coord.wf.demo")
    assert model["organisation"]["individuals"] == {"cell": ["sender", "receiver"]}
    assert model["realisation"]["parameters"]["expression"]["plan_indices"] == {
        "sender": [1, 1, 2, 2],
        "receiver": [1, 2, 1, 2],
    }
    assert model["adaptation"]["parameters"]["transmission"]["recombination"] == {
        "kind": "clonal"
    }


def test_nested_meta_resumes_between_complete_outer_tasks():
    from self_assembling_games.domain import model
    from self_assembling_games.system import System

    system = model("meta-rule.demo").instantiate(seed=31)
    system.advance(2)
    snapshot = system.snapshot()
    inner = system.inspect()["mechanisms"]["inner_task"]
    assert inner["checkpoint_granularity"] == "completed_outer_encounter"
    assert (
        inner["initialization"]
        == "fresh learner and game state for each outer encounter"
    )
    assert snapshot["state"]["runtime"]["clocks"]["problem"] == 2
    assert set(snapshot["state"]["interaction"]["last_inner"]) == {
        "seed",
        "rule",
        "accuracy",
    }
    restored = System.restore(snapshot)
    restored.advance(2)
    system.advance(2)
    assert system.snapshot() == restored.snapshot()


def test_canonical_system_large_sampled_lifecycle_never_enumerates(monkeypatch):
    from self_assembling_games.domain import model
    from self_assembling_games.system import System

    declaration = resolve_model(
        "sk-sig.r1.demo",
        {
            "interaction": {
                "parameters": {
                    "states": 8,
                    "signals": 8,
                    "actions": 8,
                    "state_probabilities": [0.125] * 8,
                }
            }
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampled System attempted exact enumeration")

    monkeypatch.setattr(FiniteGame, "_enumerate_plans", forbidden)
    monkeypatch.setattr(FiniteGame, "compile", forbidden)
    system = model(declaration).instantiate(seed=5)
    system.advance(3)
    assert system.evaluate(protocol="sampled", samples=10)["samples"] == 10
    restored = System.restore(system.snapshot())
    restored.advance(2)
    system.advance(2)
    assert restored.snapshot() == system.snapshot()
    assert (
        system.inspect()["mechanisms"]["checkpoint_granularity"]
        == "completed_encounter"
    )


@pytest.mark.parametrize(
    "preset",
    ["sig.demo", "sk-sig.r1.demo", "sk-sig.evolution.demo", "ml-coord.wf.demo"],
)
def test_zero_budget_constructs_initial_state_and_zero_advance_preserves_it(preset):
    declaration = resolve_model(preset, {"execution": {"budget": 0}})
    assert declaration["execution"]["budget"] == 0
    backend = create_backend(declaration, 1, {})
    before = backend.snapshot()
    backend.advance(0)
    assert backend.snapshot() == before


def test_raw_numerical_state_must_match_public_scientific_model():
    declaration = resolve_model("sk-sig.r1.demo")
    backend = create_backend(declaration, 2, {})
    changed = resolve_model(
        "sk-sig.r1.demo", {"realisation": {"parameters": {"initial_weight": 2.0}}}
    )
    with pytest.raises(ValueError, match="scientific model differs: realisation"):
        restore_backend(changed, backend.snapshot())


def test_restore_preserves_execution_overrides_separate_from_science():
    declaration = resolve_model("sk-sig.evolution.demo")
    backend = create_backend(declaration, 2, {"rtol": 1e-7, "sample_interval": 0.5})
    assert (
        restore_backend(declaration, backend.snapshot()).snapshot()
        == backend.snapshot()
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"resources": {"extensions": {"arbitrary": 1}}}, "Unknown Barrett resource"),
        ({"interaction": {"provider": "other"}}, "Barrett interaction"),
        ({"realisation": {"extra": 1}}, "Unknown realisation fields"),
        (
            {"organisation": {"functional_roles": {"sender": "changed"}}},
            "Functional roles",
        ),
        ({"organisation": {"owners": {"ghost": "one"}}}, "unknown module owner"),
    ],
)
def test_unsupported_native_fields_fail_explicitly(changes, match):
    with pytest.raises(ValueError, match=match):
        resolve_model("sig.demo", changes)
