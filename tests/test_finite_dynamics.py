import copy
import itertools
import json

import numpy as np
import pytest

from self_assembling_games import (
    AdaptiveProfile,
    FiniteDecision,
    FiniteGame,
    ParticipantOutcome,
    adaptive_method,
    adaptive_methods,
    adaptive_profiles,
    barrett_game,
    build,
    build_adaptive,
    evaluate,
    finite_catalogue,
    load_profile,
    register_finite_game,
    restore_adaptive,
    resume,
    skyrms_game,
)
from self_assembling_games.cli import main


def point(count, index):
    x = np.zeros(count)
    x[index] = 1
    return x


def test_evolved_and_learned_policies_share_exact_evaluation():
    learned = build_adaptive("sk-sig.strategy-r1.demo", seed=7)
    learned.run(plays=100)
    evolved = build_adaptive("sk-sig.evolution.demo", seed=8)
    for role, weights in learned.weights.items():
        evolved.set_population(role, weights / weights.sum())
    assert evolved.game_hash == learned.game_hash
    assert evolved.evaluate()["success_probability"] == pytest.approx(
        learned.evaluate()["success_probability"]
    )
    assert evolved.evaluate()["expected_returns"] == pytest.approx(
        learned.evaluate()["expected_returns"]
    )


def test_translator_evolution_matches_closed_form():
    simulation = build_adaptive("sk-chain.translator.demo", seed=17)
    initial = simulation.frequencies["translator"].copy()
    fitness = np.array([0.5, 0, 1, 0.5])
    simulation.run(duration=3)
    expected = initial * np.exp(3 * fitness)
    expected /= expected.sum()
    assert simulation.frequencies["translator"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("solver", ["DOP853", "RK45", "RK23"])
def test_supported_solvers_preserve_simplex(solver):
    simulation = build_adaptive(
        AdaptiveProfile("solver", "SK-SIG", protocol={"solver": solver}), seed=1
    )
    simulation.run(duration=0.1)
    for frequencies in simulation.frequencies.values():
        assert frequencies.sum() == pytest.approx(1)
        assert np.all(frequencies >= 0)


@pytest.mark.parametrize("solver", ["Radau", "BDF", "LSODA"])
def test_implicit_solvers_require_compatible_simplex_implementation(solver):
    with pytest.raises(ValueError, match="tangent-coordinate"):
        build_adaptive(AdaptiveProfile("solver", "SK-SIG", protocol={"solver": solver}))


def test_benchmarks_report_method_clock():
    from self_assembling_games import benchmark

    assert (
        benchmark("sk-sig.evolution.demo", duration=0.1)["clock"] == "evolutionary_time"
    )
    assert benchmark("sk-sig.r1.demo", plays=10)["clock"] == "plays"


@pytest.mark.parametrize("method", ["evolution", "strategy-r1"])
def test_adaptive_ensemble_resumes_and_keeps_native_clock(method, tmp_path):
    from self_assembling_games import run_ensemble

    kwargs = {"duration": 0.2} if method == "evolution" else {"plays": 10}
    profile = f"sk-sig.{method}.demo"
    first = run_ensemble(
        profile,
        [1, 2],
        output=tmp_path,
        **kwargs,
        metric="final_policy_success_probability",
        threshold=0.5,
    )
    second = run_ensemble(
        profile,
        [1, 2],
        output=tmp_path,
        **kwargs,
        metric="final_policy_success_probability",
        threshold=0.5,
    )
    assert first == second
    assert first["complete_runs"] == 2
    assert first["protocol"]["clock"] == (
        "evolutionary_time" if method == "evolution" else "plays"
    )


def test_frozen_import_retains_binding_and_lineage(tmp_path):
    from self_assembling_games import import_frozen_decisions

    learned = build_adaptive("sk-sig.r1.demo", seed=3)
    learned.run(plays=20, output=tmp_path)
    imported = import_frozen_decisions(
        tmp_path,
        {"sender/state:0": "sender/state:1"},
        option_indices={"sender/state:0": [1, 0]},
    )
    original = learned.weights["sender/state:1"]
    assert imported["fixed_decisions"]["sender/state:0"] == pytest.approx(
        (original / original.sum())[::-1]
    )
    assert imported["lineage"]["training_plays"] == 20
    frozen = build_adaptive(AdaptiveProfile("imported", "SK-SIG", **imported))
    assert frozen.profile.lineage == imported["lineage"]


def test_adaptive_ensemble_parallel_matches_serial(tmp_path):
    from self_assembling_games import run_ensemble

    kwargs = dict(plays=15, metric="final_policy_success_probability")
    serial = run_ensemble(
        "sk-sig.strategy-r1.demo", [2, 3], output=tmp_path / "serial", **kwargs
    )
    parallel = run_ensemble(
        "sk-sig.strategy-r1.demo",
        [2, 3],
        output=tmp_path / "parallel",
        workers=2,
        **kwargs,
    )
    assert serial == parallel


def test_ensemble_retains_resource_limited_run(tmp_path):
    from self_assembling_games import run_ensemble

    profile = AdaptiveProfile(
        "budgeted", "SK-SIG", resources={"max_field_evaluations": 1}
    )
    report = run_ensemble(profile, [2], output=tmp_path, duration=1)
    assert report["incomplete_runs"] == 1
    assert report["mean_measurement"] is None
    assert resume(tmp_path / "seed-2").status == "resource_limit"


def test_barrett_adapter_credits_only_adaptive_decisions():
    game = barrett_game("nand.demo")
    row = game.decisions[0]
    address = row.role + "/" + row.context
    simulation = build_adaptive(
        AdaptiveProfile(
            "frozen-nand",
            "NAND",
            {"profile": "nand.demo"},
            method="reinforcement.conditional.R1",
            fixed_decisions={address: [1.0] + [0.0] * (len(row.options) - 1)},
        )
    )
    simulation.run(plays=20)
    assert address not in simulation.weights
    assert simulation.evaluate()["completed_mass"] == pytest.approx(1)


@pytest.mark.parametrize(
    "parameters",
    [
        {"background": 2},
        {"signal_costs": [[0, 0], [0, 0]]},
        {"success_matrix": [[1, 0], [0, 1]]},
    ],
)
def test_custom_sig_parameters_are_explicit_variants(parameters):
    assert (
        skyrms_game("SK-SIG", parameters).fidelity["game"]["status"]
        == "explicit_parameter_variant"
    )
    with pytest.raises(ValueError, match="deliberate_variant"):
        build_adaptive(
            AdaptiveProfile("variant", "SK-SIG", parameters, kind="paper_game")
        )


@pytest.mark.parametrize(
    "model,counts",
    [
        ("SK-SIG", [4, 4]),
        ("SK-COMP", [4, 4, 256]),
        ("SK-VOTE", [4, 4, 4, 16]),
        ("SK-TEAM", [256, 4, 4]),
        ("SK-CHAIN", [4, 4, 4]),
    ],
)
def test_source_strategy_spaces(model, counts):
    game = skyrms_game(model)
    assert [len(game.plans[r]) for r in game.roles] == counts
    assert len(barrett_game("cod.demo").plans["A"]) == 16


def test_majority_conventions_and_independent_observation_errors():
    game = skyrms_game("SK-VOTE")
    frequencies = {r: point(4, 1) for r in ["A", "B", "C"]}
    frequencies["receiver"] = point(16, 3)  # count 0,1 ->0; 2,3 ->1
    assert game.expectations(frequencies)["success_probability"] == pytest.approx(0.972)
    for r in ["A", "B", "C"]:
        frequencies[r] = point(4, 2)
    frequencies["receiver"] = point(16, 12)  # reverse convention
    assert game.expectations(frequencies)["success_probability"] == pytest.approx(0.972)
    # A constant sender's signal is reliable even though its observation is noisy.
    for r in ["A", "B", "C"]:
        frequencies[r] = point(4, 0)
    assert game.expectations(frequencies)["success_probability"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "model,indices",
    [
        ("SK-COMP", [1, 1, 27]),
        ("SK-TEAM", [27, 1, 1]),
        ("SK-CHAIN", [1, 2, 2]),
        ("SK-SIG", [1, 1]),
    ],
)
def test_perfect_source_fixtures(model, indices):
    game = skyrms_game(model)
    frequencies = {
        r: point(len(game.plans[r]), index) for r, index in zip(game.roles, indices)
    }
    result = game.expectations(frequencies)
    assert result["success_probability"] == pytest.approx(1)
    assert result["completed_mass"] == pytest.approx(1)
    assert all(
        value == pytest.approx(1) for value in result["expected_returns"].values()
    )


def test_shared_population_fitness_matches_role_averaged_encounters():
    simulation = build_adaptive("sk-sig.shared.demo", seed=9)
    types = list(itertools.product(range(4), repeat=2))
    plans = list(itertools.product(range(2), repeat=2))
    matrix = np.empty((16, 16))
    for i, (sender_i, receiver_i) in enumerate(types):
        for j, (sender_j, receiver_j) in enumerate(types):
            matrix[i, j] = (
                sum(
                    (plans[receiver_j][plans[sender_i][state]] == state)
                    + (plans[receiver_i][plans[sender_j][state]] == state)
                    for state in [0, 1]
                )
                / 4
            )
    x = simulation.frequencies["population"]
    fitness = matrix @ x
    assert simulation.vector_field()["population"] == pytest.approx(
        x * (fitness - x @ fitness)
    )


def test_grouped_population_lifts_two_distinct_sender_plans():
    simulation = build_adaptive("sk-comp.grouped.demo", seed=4)
    sender_plans = list(itertools.product(range(2), repeat=2))
    receiver_plans = list(itertools.product(range(4), repeat=4))
    tensor = np.zeros((4, 4, 256))
    for a, b, r in itertools.product(range(4), range(4), range(256)):
        tensor[a, b, r] = (
            sum(
                receiver_plans[r][2 * sender_plans[a][s // 2] + sender_plans[b][s % 2]]
                == s
                for s in range(4)
            )
            / 4
        )
    x = simulation.frequencies["senders"].reshape(4, 4)
    y = simulation.frequencies["receivers"]
    fa = np.einsum("abr,b,r->a", tensor, x.sum(axis=0), y)
    fb = np.einsum("abr,a,r->b", tensor, x.sum(axis=1), y)
    expected = ((fa[:, None] + fb[None, :]) / 2).ravel()
    actual = simulation._expectations(simulation.frequencies)["population_fitness"][
        "senders"
    ]
    assert actual == pytest.approx(expected)
    assert simulation.population.counts == {"receivers": 256, "senders": 16}


def test_complete_plan_correlations_survive_leaf_contraction():
    def interaction(provider):
        a = provider.decide("animal", "first")
        b = provider.decide("animal", "second")
        return ParticipantOutcome(a == b, (float(a == b),))

    game = FiniteGame(
        "CORRELATION",
        ("animal",),
        (
            FiniteDecision("animal", "first", (0, 1), "A"),
            FiniteDecision("animal", "second", (0, 1), "B"),
        ),
        interaction,
    )
    correlated = np.array([0.5, 0, 0, 0.5])
    anticorrelated = np.array([0, 0.5, 0.5, 0])
    assert game.expectations({"animal": correlated})["success_probability"] == 1
    assert game.expectations({"animal": anticorrelated})["success_probability"] == 0
    assert (
        game.conditional_expectations(
            {
                "animal/first": np.array([0.5, 0.5]),
                "animal/second": np.array([0.5, 0.5]),
            }
        )["success_probability"]
        == 0.5
    )


def test_listening_fee_belongs_to_complete_plan_and_discrepancy_is_visible():
    prose = skyrms_game(
        "SK-SIG", {"cost_case": "listening", "cost_interpretation": "prose_derived"}
    )
    literal = skyrms_game(
        "SK-SIG", {"cost_case": "listening", "cost_interpretation": "table_literal"}
    )
    # Always send first; receiver's identity mapping chooses act0.
    x = {"sender": point(4, 0), "receiver": point(4, 1)}
    assert prose.expectations(x)["expected_returns"] == pytest.approx(
        {"sender": 4 / 3 - 0.3, "receiver": 4 / 3 - 0.1}
    )
    assert literal.expectations(x)["expected_returns"] == pytest.approx(
        {"sender": 1.5 - 0.3, "receiver": 1.5 - 0.1}
    )
    # An unconditional receiver taking the same realized act pays no listening fee.
    x["receiver"] = point(4, 0)
    assert prose.expectations(x)["expected_returns"]["receiver"] == pytest.approx(4 / 3)
    assert literal.expectations(x)["expected_returns"]["receiver"] == pytest.approx(
        1.33
    )


def test_conditional_listening_cost_is_source_gated():
    p = AdaptiveProfile(
        "listening-r1",
        "SK-SIG",
        {"cost_case": "listening", "cost_interpretation": "prose_derived"},
        method="reinforcement.conditional.R1",
    )
    with pytest.raises(ValueError, match="SKQ3"):
        build_adaptive(p)


def test_source_game_and_historical_protocol_gates():
    with pytest.raises(ValueError, match="SKQ4"):
        skyrms_game("SK-DIALOGUE")
    with pytest.raises(ValueError, match="SKQ2"):
        skyrms_game("SK-SIG", {"cost_case": "listening"})
    p = AdaptiveProfile(
        "dialogue-source",
        "SK-DIALOGUE",
        {"state_probabilities": [0.25] * 4, "reply_binding": "partition_cell"},
        kind="paper_game",
    )
    with pytest.raises(ValueError, match="SKQ4"):
        build_adaptive(p)
    with pytest.raises(ValueError, match="SKQ1"):
        build_adaptive(
            AdaptiveProfile("history", "SK-SIG", kind="historical_reproduction")
        )


@pytest.mark.parametrize("binding", ["partition_cell", "query_cell"])
def test_dialogue_timing_and_perfect_convention(binding):
    game = skyrms_game(
        "SK-DIALOGUE", {"state_probabilities": [0.25] * 4, "reply_binding": binding}
    )
    assert [len(v) for v in game.plans.values()] == [64, 64]
    # Partition selector, replies, query, decoder: identity binary maps.
    plans = {r: game.plans[r] for r in game.roles}
    target = [0, 1, 0, 1, 0, 1]
    frequencies = {
        r: point(len(p), int(np.flatnonzero(np.all(p == target, axis=1))[0]))
        for r, p in plans.items()
    }
    assert game.expectations(frequencies)["success_probability"] == pytest.approx(1)
    assert game.expectations(frequencies)["expected_exchanges"] == pytest.approx(1)


def test_dialogue_encodings_have_distinct_off_equilibrium_behavior():
    # Both queries select partition0, but player1 has different reply maps.
    frequencies = {"observer": point(64, 6), "decider": point(64, 37)}
    values = []
    for binding in ["partition_cell", "query_cell"]:
        game = skyrms_game(
            "SK-DIALOGUE", {"state_probabilities": [0.25] * 4, "reply_binding": binding}
        )
        values.append(game.expectations(frequencies)["success_probability"])
    assert values[0] != values[1]


def test_birth_mutation_matrix_and_signed_domain():
    x = np.array([0.2, 0.3, 0.5, 0])
    f = np.array([0.5, 1, 0.4, 0.9])
    epsilon = 0.02
    matrix = (1 - epsilon) * np.eye(4) + epsilon * np.ones((4, 4)) / 4
    method = adaptive_method("replicator_mutator.birth_uniform")
    assert method.field(x, f, epsilon) == pytest.approx((x * f) @ matrix - x * (x @ f))
    assert method.field(x, f, 0) == pytest.approx(
        adaptive_method("replicator.continuous").field(x, f)
    )
    assert method.field(x, np.zeros(4), epsilon) == pytest.approx(np.zeros(4))
    assert method.field(x, f, epsilon)[-1] > 0
    with pytest.raises(ValueError, match="nonnegative"):
        method.field(x, f - 1, epsilon)


def test_background_invariance_is_specific_to_pure_selection():
    x = np.array([0.2, 0.3, 0.5])
    f = np.array([0.4, 0.6, 0.8])
    pure = adaptive_method("replicator.continuous")
    mutation = adaptive_method("replicator_mutator.birth_uniform")
    assert pure.field(x, f) == pytest.approx(pure.field(x, f + 1))
    assert not np.allclose(mutation.field(x, f, 0.1), mutation.field(x, f + 1, 0.1))


def test_stationary_invader_diagnostics_and_uniform_symmetry():
    s = build_adaptive("sk-sig.evolution.demo")
    s.set_population("sender", point(4, 2))  # inverted sender
    s.set_population("receiver", point(4, 1))  # identity receiver: all fail
    result = s.evaluate()
    assert max(result["stationarity_residual"].values()) == 0
    assert result["absent_strategy_invasion_advantage"]["sender"] == 1
    assert not result["stability_assessed"]
    symmetric = build_adaptive(
        AdaptiveProfile(
            "uniform-fixture",
            "SK-SIG",
            protocol={"initialization": {"kind": "uniform"}},
        )
    )
    before = copy.deepcopy(symmetric.frequencies)
    symmetric.run(duration=2)
    for r in before:
        assert symmetric.frequencies[r] == pytest.approx(before[r])


@pytest.mark.parametrize(
    "method,epsilon",
    [("replicator.continuous", 0), ("replicator_mutator.birth_uniform", 0.1)],
)
def test_integration_conservation_zero_support_and_resume(method, epsilon, tmp_path):
    p = AdaptiveProfile(
        "ode-test", "SK-SIG", method=method, protocol={"mutation_rate": epsilon}
    )
    s = build_adaptive(p, seed=3)
    s.set_population("sender", [0.4, 0.6, 0, 0])
    s.run(duration=0.5)
    loaded = resume(s.checkpoint(tmp_path / method))
    s.run(duration=0.5)
    loaded.run(duration=0.5)
    assert loaded.snapshot() == s.snapshot()
    assert s.frequencies["sender"].sum() == pytest.approx(1, abs=1e-12)
    if epsilon == 0:
        assert np.array_equal(s.frequencies["sender"][2:], [0, 0])
    else:
        assert np.all(s.frequencies["sender"][2:] > 0)


def test_solver_tolerance_and_numerical_segment_agreement():
    p = AdaptiveProfile("tolerance", "SK-SIG")
    ordinary = build_adaptive(p, seed=10)
    p.protocol = {"rtol": 1e-11, "atol": 1e-14}
    tight = build_adaptive(p, seed=10)
    segmented = build_adaptive(p, seed=10)
    ordinary.run(duration=3)
    tight.run(duration=3)
    segmented.run(duration=1.5)
    segmented.run(duration=1.5)
    for name in tight.frequencies:
        assert ordinary.frequencies[name] == pytest.approx(
            tight.frequencies[name], abs=1e-9
        )
        assert segmented.frequencies[name] == pytest.approx(
            tight.frequencies[name], abs=1e-9
        )
    assert ordinary.summary()["time_average_expected_success"] == pytest.approx(
        tight.summary()["time_average_expected_success"], abs=1e-9
    )


def test_frozen_translation_has_unique_optimal_plan():
    s = build_adaptive("sk-chain.translator.demo")
    assert len(s.game.plans["sender"]) == len(s.game.plans["receiver"]) == 1
    f = s._expectations(s.frequencies)["fitness"]["translator"]
    assert f == pytest.approx([0.5, 0, 1, 0.5])
    assert s.vector_field()["sender"] == pytest.approx([0])


@pytest.mark.parametrize(
    "method", ["reinforcement.conditional.R1", "reinforcement.strategy.R1"]
)
def test_learning_checkpoint_and_evaluation_isolation(method, tmp_path):
    s = build_adaptive(AdaptiveProfile("learning", "SK-COMP", method=method), seed=11)
    s.run(plays=25)
    before = s.snapshot()
    s.evaluate()
    s.evaluate(protocol="sampled", samples=20, seed=200)
    assert s.snapshot() == before
    assert evaluate(s)["completed_mass"] == pytest.approx(1)
    restored = resume(s.checkpoint(tmp_path / method))
    s.run(plays=10)
    restored.run(plays=10)
    assert restored.snapshot() == s.snapshot()


def test_strategy_credit_is_once_per_participant_and_background_survives_failure():
    p = AdaptiveProfile(
        "strategy-credit",
        "SK-DIALOGUE",
        {"reply_binding": "partition_cell", "state_probabilities": [0.25] * 4},
        method="reinforcement.strategy.R1",
    )
    s = build_adaptive(p)
    before = {r: w.sum() for r, w in s.weights.items()}
    result = s.step()
    for role in s.game.roles:
        assert s.weights[role].sum() - before[role] == result.success
    p = AdaptiveProfile(
        "background", "SK-SIG", {"background": 1}, method="reinforcement.strategy.R1"
    )
    s = build_adaptive(p)
    s.set_policy("sender", point(4, 2), frozen=False)
    s.set_policy("receiver", point(4, 1), frozen=False)
    before = {r: w.sum() for r, w in s.weights.items()}
    result = s.step()
    assert not result.success
    for role in s.game.roles:
        assert s.weights[role].sum() - before[role] == 1


def test_barrett_adapter_preserves_source_r1_trajectory_and_credit():
    old = build("nand.demo", seed=13)
    adapted = build_adaptive(
        AdaptiveProfile(
            "adapted",
            "NAND",
            {"profile": "nand.demo"},
            method="reinforcement.conditional.R1",
        ),
        seed=13,
    )
    old.run(plays=40)
    adapted.run(plays=40)
    for role, rows in adapted.game.rows.items():
        for row in rows:
            assert adapted.weights[role + "/" + row.context] == pytest.approx(
                old.runtime.tables[row.context].weights
            )
    assert (
        adapted.summary()["cumulative_accuracy"]
        == old.metrics.summary()["cumulative_accuracy"]
    )


def test_generic_game_registration_uses_existing_method_classes(tmp_path):
    def factory(parameters, fixed_decisions, **budgets):
        def interaction(provider):
            act = provider.decide("animal", "choice")
            return ParticipantOutcome(act == 1, (float(act),))

        return FiniteGame(
            "TEST-BANDIT",
            ("animal",),
            (FiniteDecision("animal", "choice", (0, 1), "actor"),),
            interaction,
            parameters,
            {"game": {"status": "test_fixture"}},
            fixed_decisions,
            **budgets,
        )

    register_finite_game(
        "TEST-BANDIT",
        factory,
        family="test",
        description="One-decision fixture",
        source="test",
    )
    p = AdaptiveProfile("generic", "TEST-BANDIT")
    s = build_adaptive(p)
    x = s.frequencies["animal"]
    assert s.vector_field()["animal"] == pytest.approx(x * (np.array([0, 1]) - x[1]))
    assert resume(s.checkpoint(tmp_path / "generic")).snapshot() == s.snapshot()


def test_resource_and_configuration_failures_are_explicit():
    with pytest.raises(ValueError, match="max_role_strategies"):
        build_adaptive(AdaptiveProfile("large", "SK-SIG", {"states": 8}))
    with pytest.raises(ValueError, match="max_population_strategies"):
        build_adaptive(
            AdaptiveProfile(
                "large-comp",
                "SK-COMP",
                population={"groups": {"all": ["A", "B", "receiver"]}},
                resources={"max_population_strategies": 100},
            )
        )
    with pytest.raises(ValueError, match="mask"):
        build_adaptive(
            AdaptiveProfile(
                "mask-budget", "SK-COMP", resources={"max_mask_entries": 10}
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_adaptive(
            AdaptiveProfile(
                "bad-pop",
                "SK-COMP",
                population={"groups": {"A": ["A", "B"], "R": ["B", "receiver"]}},
            )
        )
    with pytest.raises(ValueError, match="stationary finite"):
        barrett_game("inv.demo")
    with pytest.raises(ValueError, match="no implicit payoff shift"):
        build_adaptive(
            AdaptiveProfile(
                "negative",
                "SK-SIG",
                {"background": -0.1},
                method="reinforcement.conditional.R1",
            )
        )


@pytest.mark.parametrize("profile", adaptive_profiles(), ids=lambda p: p["id"])
def test_every_adaptive_profile_has_exact_mass_and_restores(profile):
    s = build_adaptive(profile, seed=2)
    assert s.evaluate()["completed_mass"] == pytest.approx(1)
    if s.method.method_class == "reinforcement":
        s.run(plays=2)
    elif s.method.method_class == "population_generations":
        s.run(generations=2)
    else:
        s.run(duration=0.01)
    assert restore_adaptive(s.snapshot()).snapshot() == s.snapshot()


def test_adaptive_cli_and_family_listings(tmp_path, capsys):
    assert load_profile("sk-sig.evolution.demo").method == "replicator.continuous"
    assert len(finite_catalogue("skyrms")) == 6
    assert {m["method_class"] for m in adaptive_methods()} == {
        "population_evolution",
        "reinforcement",
        "population_generations",
    }
    assert main(["methods"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6
    assert main(["list", "--all", "--family", "skyrms"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6
    output = tmp_path / "evolution"
    assert (
        main(["run", "sk-sig.evolution.demo", "--time", ".1", "--output", str(output)])
        == 0
    )
    capsys.readouterr()
    assert main(["resume", str(output), "--time", ".1"]) == 0
    capsys.readouterr()
    assert resume(output).time == pytest.approx(0.2)
    assert main(["evaluate", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "exact_finite"
    assert (
        main(
            [
                "run",
                "sk-sig.evolution.demo",
                "--plays",
                "10",
                "--output",
                str(tmp_path / "bad"),
            ]
        )
        == 2
    )
    assert "--time" in capsys.readouterr().err


def test_restoration_checks_semantics_and_protocol():
    s = build_adaptive("sk-sig.evolution.demo")
    snapshot = s.snapshot()
    snapshot["game"]["parameters"]["states"] = 3
    with pytest.raises(ValueError, match="specification"):
        restore_adaptive(snapshot)
    snapshot = s.snapshot()
    snapshot["protocol"]["max_step"] = 99
    with pytest.raises(ValueError, match="protocol"):
        restore_adaptive(snapshot)
