"""Analytic fields, local feedback domains, and terminal reproductive censuses."""

import copy

import numpy as np
import pytest

from self_assembling_games import (
    AdaptiveProfile,
    FiniteDecision,
    FiniteGame,
    ParticipantOutcome,
    adaptive_method,
    build_adaptive,
    load_adaptive_profile,
    resume,
)
from self_assembling_games.organisation import Organisation
from self_assembling_games.realisations import (
    FiniteReinforcementRealisation,
    validate_finite_realisation,
)


def local_game(owner_return=1.0, credit=1.0):
    def encounter(provider):
        option = provider.decide("animal", "choice")
        provider.credits = [("animal", "choice", option, credit)]
        return ParticipantOutcome(True, (owner_return,))

    return FiniteGame(
        "LOCAL-RETURNS",
        ("animal",),
        (FiniteDecision("animal", "choice", (0, 1), "actor"),),
        encounter,
    )


@pytest.mark.parametrize("background", [-1, -10, -100, -1000])
def test_signed_background_preserves_pure_replicator_policy(background):
    baseline, shifted = (
        build_adaptive(
            AdaptiveProfile("background", "SK-SIG", {"background": value}), seed=3
        )
        for value in (0, background)
    )
    baseline.run(duration=3)
    shifted.run(duration=3)
    for role in baseline.game.roles:
        np.testing.assert_allclose(
            shifted.frequencies[role], baseline.frequencies[role], atol=1e-9, rtol=0
        )
        assert shifted.evaluate()["expected_returns"][role] == pytest.approx(
            baseline.evaluate()["expected_returns"][role] + background, abs=1e-9
        )
    assert shifted.history[-1]["field_evaluations"] < 1000


def test_payoff_gauge_and_positive_scaling_respect_support_and_birth_mutation():
    x = np.array([0.2, 0.3, 0.5, 0])
    f = np.array([0.4, 0.6, 0.8, 1.0])
    pure = adaptive_method("replicator.continuous")
    birth = adaptive_method("replicator_mutator.birth_uniform")
    expected = x * (f - x @ f)
    np.testing.assert_allclose(pure.field(x, f - 1000), expected, atol=1e-13)
    np.testing.assert_allclose(pure.field(x, 3 * f), 3 * expected, atol=1e-15)
    assert pure.field(x, f - 1000)[-1] == 0
    assert sum(pure.field(x, f - 1000)) == pytest.approx(0, abs=1e-15)
    epsilon = 0.1
    np.testing.assert_allclose(
        birth.field(x, f + 1, epsilon) - birth.field(x, f, epsilon),
        epsilon * (1 / len(x) - x),
        atol=1e-15,
    )
    assert birth.field(x, f, epsilon)[-1] > 0


def test_conditional_r1_uses_local_credit_instead_of_owner_return():
    method = adaptive_method("reinforcement.conditional.R1")
    game = local_game(owner_return=-2, credit=0.5)
    realisation = FiniteReinforcementRealisation(
        game, Organisation.for_finite(game), method, {"initial_weight": 1}, seed=1
    )
    outcome = realisation.encounter(game)
    assert outcome.returns == (-2,)
    assert realisation.weights["animal/choice"].sum() == 2.5
    with pytest.raises(ValueError, match="reinforcement credit"):
        validate_finite_realisation(local_game(1, -0.5), method)
    with pytest.raises(ValueError, match="no implicit payoff shift"):
        validate_finite_realisation(game, adaptive_method("reinforcement.strategy.R1"))
    with pytest.raises(ValueError, match="reproduction returns"):
        validate_finite_realisation(
            game, adaptive_method("replicator_mutator.birth_uniform")
        )


def test_fixed_and_frozen_rows_receive_no_negative_local_credit():
    method = adaptive_method("reinforcement.conditional.R1")
    game = local_game(1, -1)
    game.fixed_decisions = {"animal/choice": [1, 0]}
    game.__post_init__()
    validate_finite_realisation(game, method)
    realisation = FiniteReinforcementRealisation(
        game, Organisation.for_finite(game), method, {"initial_weight": 1}
    )
    realisation.encounter(game)
    assert realisation.weights == {}

    game = local_game(1, 1)
    realisation = FiniteReinforcementRealisation(
        game, Organisation.for_finite(game), method, {"initial_weight": 1}
    )
    realisation.frozen.add("animal/choice")
    before = realisation.weights["animal/choice"].copy()
    # A live intervention can stop adaptation after construction.
    game.interaction = local_game(1, -1).interaction
    realisation.encounter(game)
    np.testing.assert_array_equal(realisation.weights["animal/choice"], before)


def test_sampled_strategy_cost_gate_checks_arrays_and_deferred_capability():
    method = adaptive_method("reinforcement.conditional.R1")
    game = local_game()
    game.strategy_costs["animal"] = np.array([0.0, 0.5])
    with pytest.raises(ValueError, match="SKQ3"):
        validate_finite_realisation(game, method, sampled=True)

    calls = []
    game = local_game()
    game.strategy_costs.defer("animal", lambda: calls.append(True) or np.zeros(2))
    with pytest.raises(ValueError, match="SKQ3"):
        validate_finite_realisation(game, method, sampled=True)
    assert not calls
    assert not game.plans.cache

    game.strategy_costs.defer(
        "animal", lambda: calls.append(True) or np.zeros(2), known_zero=True
    )
    validate_finite_realisation(game, method, sampled=True)
    assert not calls
    # Cached arrays remain mutable, so the cheap gate also checks cached values.
    game.strategy_costs["animal"][1] = 0.5
    with pytest.raises(ValueError, match="SKQ3"):
        validate_finite_realisation(game, method, sampled=True)


def test_exact_one_step_r1_drift_keeps_reward_distribution_information():
    def encounter(provider):
        option = provider.decide("animal", "choice")
        reward = 1.0 if option == 0 else 2.0 * provider.chance([0.5, 0.5])
        return ParticipantOutcome(True, (reward,))

    game = FiniteGame(
        "REWARD-VARIANCE",
        ("animal",),
        (FiniteDecision("animal", "choice", (0, 1), "actor"),),
        encounter,
    )
    game.compile()
    weights = np.ones(2)
    drift = np.zeros(2)
    for chance, draws, returns in zip(game.chance_mass, game.draw_paths, game.returns):
        option = draws[0][2]
        updated = weights.copy()
        updated[option] += returns[0]
        drift += chance * 0.5 * (updated / updated.sum() - 0.5)
    np.testing.assert_allclose(drift, [1 / 48, -1 / 48], atol=1e-15)
    np.testing.assert_array_equal(
        adaptive_method("replicator.continuous").field([0.5, 0.5], [1, 1]), [0, 0]
    )


@pytest.mark.parametrize("finite", [False, True])
def test_valid_zero_reproduction_next_census_is_retained_and_resumes(finite, tmp_path):
    profile = load_adaptive_profile(
        f"ml-coord.{'wf' if finite else 'deterministic'}.demo"
    )
    profile.genetics["fitness"] = {"background": 0, "scale": 1}
    profile.genetics["transmission"] = {
        "recombination": {"kind": "clonal"},
        "mutation": {
            "s": [[1, 0], [1, 0]],
            "r": [[0, 1], [0, 1]],
        },
    }
    if finite:
        profile.genetics["population_size"] = 4
    profile.protocol["initialization"] = {
        "kind": "explicit_counts" if finite else "frequencies",
        "counts" if finite else "frequencies": [2, 0, 0, 2]
        if finite
        else [0.5, 0, 0, 0.5],
    }
    simulation = build_adaptive(profile, seed=11)
    assert simulation.evaluate()["mean_reproduction_fitness"] == 1
    report = simulation.run(generations=3, output=tmp_path)
    assert simulation.generation == 1
    assert report["status"] == "no_reproduction"
    assert report["mean_reproduction_fitness"] == 0
    assert report["generation_average_expected_success"] == 1
    np.testing.assert_array_equal(simulation.frequencies, [0, 1, 0, 0])
    assert simulation.snapshot()["last_generation"]["after_census"] == [0, 1, 0, 0]
    stopped = copy.deepcopy(simulation.snapshot())
    restored = resume(tmp_path)
    assert restored.snapshot() == stopped
    assert restored.step() == {"status": "no_reproduction", "generation": 1}
    restored.run(generations=3)
    assert restored.snapshot() == stopped
    restored.set_population([2, 0, 0, 2] if finite else [0.5, 0, 0, 0.5])
    assert restored.status == "ready"
    restored.step()
    assert restored.generation == 2
    assert restored.status == "no_reproduction"
