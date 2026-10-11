"""Executed distinctions and compatibility of the research-facing domain model."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import self_assembling_games as sg
from self_assembling_games.cli import main
from self_assembling_games.simulation import digest


PROFILES = sg.profiles() + sg.adaptive_profiles()


def source(specification):
    declaration = specification.to_dict()["source"]
    return sg.SourceDeclaration(**declaration)


def native(specification):
    return sg.ModelSpecification.resolve(
        game=specification.game_id,
        game_parameters=specification.game_parameters,
        realisation=specification.realisation_id,
        realisation_parameters=specification.realisation_parameters,
        organisation=specification.organisation_parameters,
        source=source(specification),
    )


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p["id"])
def test_native_and_profile_execute_the_same_model(profile):
    specification = sg.ModelSpecification.from_profile(profile)
    resolved = native(specification)
    assert resolved.to_dict() == specification.to_dict()
    left, right = sg.build(profile, 23), sg.build(resolved, 23)
    assert left.model_instance.inspect() == right.model_instance.inspect()
    if isinstance(left, sg.EvolutionSimulation):
        for name, field in left.vector_field().items():
            np.testing.assert_array_equal(field, right.vector_field()[name])
        left.run(duration=0.02)
        right.run(duration=0.02)
        for name in left.frequencies:
            np.testing.assert_array_equal(
                left.frequencies[name], right.frequencies[name]
            )
    else:
        for _ in range(5):
            assert left.step() == right.step()
        assert left.model_instance.inspect() == right.model_instance.inspect()


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p["id"])
def test_pre_refactor_state_rng_events_and_trajectory(profile):
    fixture = json.loads(
        (Path(__file__).parent / "data/domain_refactor_baseline.json").read_text()
    )
    simulation = sg.build(profile, fixture["seed"])
    snapshots, events = [simulation.snapshot()], []
    if isinstance(simulation, sg.Simulation):
        simulation.runtime.record = True
        for _ in range(fixture["encounter_steps"]):
            simulation.step()
            events.append(copy.deepcopy(simulation.runtime.events))
    elif isinstance(simulation, sg.EvolutionSimulation):
        simulation.run(duration=fixture["evolution_duration"])
    else:
        simulation.run(plays=fixture["finite_plays"])
    snapshots.append(simulation.snapshot())
    for snapshot in snapshots:
        for key in (
            "metadata",
            "model_specification",
            "model_specification_hash",
            "model_protocol",
            "construction_state",
        ):
            snapshot.pop(key, None)
    assert (
        digest({"snapshots": snapshots, "events": events})
        == fixture["cases"][profile["id"]]
    )


def test_equal_conditional_dispositions_different_adaptation():
    runtime = sg.Runtime()
    addresses = [
        runtime.add_table(m, "input", [0, 1], {"rule": "R1"}, initial=w)
        for m, w in [("small", 1), ("large", 100)]
    ]
    small, large = [runtime.tables[a] for a in addresses]
    np.testing.assert_array_equal(
        small.conditional_disposition(), large.conditional_disposition()
    )
    for address in addresses:
        runtime.feedback(sg.Feedback(sg.Address(address, 0), True, 1))
    assert small.conditional_disposition()[0] == pytest.approx(2 / 3)
    assert large.conditional_disposition()[0] == pytest.approx(101 / 201)


def test_specification_and_inspection_are_defensive_and_seed_independent():
    parameters = {"states": 2, "signals": 2, "actions": 2}
    learner = {"rule": "R1"}
    spec = sg.ModelSpecification.resolve(
        game="SIG",
        game_parameters=parameters,
        realisation="barrett.conditional",
        realisation_parameters=learner,
    )
    initial = spec.to_dict()
    parameters["states"] = 99
    learner["rule"] = "unknown"
    spec.game_parameters["state_probabilities"][0] = 0
    assert spec.to_dict() == initial
    assert initial["game"]["parameters"]["state_probabilities"] == [0.5, 0.5]
    assert initial["organisation"]["owners"] == {
        "sender": "sender",
        "receiver": "receiver",
    }
    assert initial["realisation"]["initial_conditions"]["table_weight"] == 1
    assert (
        sg.build(spec, 1).model_instance.specification.to_dict()
        == sg.build(spec, 2).model_instance.specification.to_dict()
    )
    simulation = sg.build(spec, 8)
    before = simulation.snapshot()
    simulation.model_instance.inspect()["realisation"]["tables"].clear()
    sg.evaluate(simulation)
    assert simulation.snapshot() == before


@pytest.mark.parametrize(
    "profile",
    [
        "sig.demo",
        "sig.strategy.demo",
        "band.arp.demo",
        "sk-sig.r1.demo",
        "sk-sig.strategy-r1.demo",
        "sk-sig.shared.demo",
    ],
)
def test_live_state_aliases_and_native_checkpoint_declaration(profile):
    spec = native(sg.ModelSpecification.from_profile(profile))
    declaration = spec.to_dict()["source"]
    declaration["identifiers"] = ["research:explicit-assay"]
    declaration["questions"] += ["RESEARCH-Q1"]
    spec = sg.ModelSpecification.resolve(
        game=spec.game_id,
        game_parameters=spec.game_parameters,
        realisation=spec.realisation_id,
        realisation_parameters=spec.realisation_parameters,
        organisation=spec.organisation_parameters,
        source=sg.SourceDeclaration(**declaration),
    )
    simulation = sg.build(spec, 12)
    model = simulation.model_instance
    if isinstance(simulation, sg.Simulation):
        assert simulation.runtime is model.realisation
        assert simulation.world is model.realisation.world
        assert simulation.runtime.owners is model.organisation.owners
        assert simulation.runtime.module_roles is model.organisation.module_roles
        restored = sg.Simulation.restore(simulation.snapshot())
        replacement = sg.Runtime.restore(simulation.runtime.state())
        simulation.runtime = replacement
        assert simulation.model_instance.realisation is replacement
    else:
        restored = sg.restore_adaptive(simulation.snapshot())
        if isinstance(simulation, sg.EvolutionSimulation):
            assert simulation.frequencies is model.realisation.frequencies
            assert simulation.population is model.organisation.population
            assert (
                restored.population is restored.model_instance.organisation.population
            )
            assert (
                restored.frequencies is restored.model_instance.realisation.frequencies
            )
        else:
            assert simulation.weights is model.realisation.weights
            simulation.weights[next(iter(simulation.weights))][0] += 3
            assert simulation.weights is model.realisation.weights
    assert restored.model_instance.specification.to_dict() == spec.to_dict()


def test_source_and_method_gates_apply_to_native_resolution():
    with pytest.raises(ValueError, match="resolution"):
        sg.ModelSpecification.resolve(
            game="SIG",
            game_parameters={"states": 2, "signals": 2, "actions": 2},
            realisation="barrett.conditional",
            realisation_parameters={"rule": "R1"},
            source=sg.SourceDeclaration(kind="source_resolved", questions=("Q-extra",)),
        )
    with pytest.raises(ValueError, match="R4"):
        sg.ModelSpecification.resolve(
            game="SIG",
            game_parameters={"states": 2, "signals": 2, "actions": 2},
            realisation="barrett.conditional",
            realisation_parameters={"rule": "R4"},
        )
    with pytest.raises(ValueError, match="SKQ3"):
        sg.ModelSpecification.resolve(
            game="SK-SIG",
            game_parameters={
                "cost_case": "listening",
                "cost_interpretation": "prose_derived",
            },
            realisation="finite.conditional",
            realisation_parameters={},
        )
    with pytest.raises(ValueError, match="disagree"):
        sg.ModelSpecification.resolve(
            game="SK-SIG",
            game_parameters={},
            realisation="finite.conditional",
            realisation_parameters={"method": "reinforcement.strategy.R1"},
        )
    with pytest.raises(ValueError, match="Outer run"):
        sg.ModelSpecification.resolve(
            game="SK-SIG",
            game_parameters={},
            realisation="finite.conditional",
            realisation_parameters={"plays": 3},
        )


def test_slot_bindings_come_from_operative_decisions_and_both_credit_paths_use_them():
    simulation = sg.build("sk-sig.r1.demo")
    organisation = simulation.model_instance.organisation
    assert organisation._game is simulation.game
    assert organisation.decision("sender", "state:0") is simulation.game.decision(
        "sender", "state:0"
    )
    # A returned binding mapping is a view of game records, never a second mutable graph.
    organisation.slots["sender"] = ("invented",)
    assert organisation.slots["sender"] == ("sender",)
    realisation = simulation.model_instance.realisation
    original = organisation.policy_address
    calls = []

    def resolve(slot, context):
        calls.append((slot, context))
        return original(slot, context)

    organisation.policy_address = resolve
    simulation.step()
    assert len(calls) == 4  # two selections and two addressed credits
    before = {k: v.copy() for k, v in simulation.weights.items()}
    provider = SimpleNamespace(
        plan_indices=None,
        credits=[("sender", "state:0", 0, 1), ("receiver", "signal:0", 0, -1)],
    )
    with pytest.raises(ValueError, match="nonnegative"):
        realisation.adapt(provider, SimpleNamespace(returns=(1, 1)))
    for key in before:
        np.testing.assert_array_equal(before[key], simulation.weights[key])


def test_sequential_barrett_failure_preserves_its_observable_partial_update():
    runtime = sg.Runtime()
    address = runtime.add_table("module", 0, [0, 1], {"rule": "R1"})
    # Use a finite starting total then overflow a single addressed update.
    table = runtime.tables[address]
    table.weights[:] = [1e308, 1]
    table.validate()
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        runtime.feedback(sg.Feedback(sg.Address(address, 0), True, 1e308))
    assert np.isinf(table.weights[0])


def test_inspection_distinguishes_propensities_from_persistent_act_and_plan_distributions():
    runtime = sg.Runtime()
    address = runtime.add_table("module", 0, [0, 1], {"rule": "WSLR"})
    table = runtime.tables[address]
    table.weights[:] = [1, 99]
    table.current = 0
    table.validate()
    state = table.inspect_dispositions()
    assert state["conditional_disposition"] == [1, 0]
    assert state["propensity_distribution"] == [0.01, 0.99]
    assert "resampling_disposition" not in state
    assert (
        "plan_selection_distributions"
        in sg.build("sk-sig.strategy-r1.demo").model_instance.inspect()["realisation"]
    )
    assert (
        "encounter_plan_distributions"
        in sg.build("sk-sig.shared.demo").model_instance.inspect()["realisation"]
    )


def test_resolved_cli_initialises_without_advancing_a_play(capsys):
    assert main(["describe", "sig.demo", "--resolved", "--seed", "11"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["realisation"]["clocks"]["play"] == 0
    assert resolved["specification"]["realisation"]["id"] == "barrett.conditional"


def test_checkpoint_model_declaration_is_checked():
    simulation = sg.build("sig.demo")
    snapshot = simulation.snapshot()
    snapshot["model_specification"]["game"]["parameters"]["states"] = 3
    snapshot["model_specification_hash"] = digest(snapshot["model_specification"])
    with pytest.raises(ValueError, match="specification differs"):
        sg.Simulation.restore(snapshot)


@pytest.mark.parametrize(
    "method,constructor",
    [
        ("reinforcement.conditional.R1", sg.ReinforcementSimulation),
        ("replicator.continuous", sg.EvolutionSimulation),
    ],
)
def test_game_overrides_are_copied_bound_and_validated_for_profiles_and_specifications(
    method, constructor
):
    profile = sg.AdaptiveProfile("override", "SK-SIG", method=method, kind="paper_game")
    specification = sg.ModelSpecification.from_profile(profile)
    game = sg.skyrms_game("SK-SIG")
    copied = sg.ModelSpecification.from_profile(profile, game=game)
    before = copied.to_dict()
    game.parameters["states"] = 99
    assert copied.to_dict() == before
    assert sg.build(copied).game.parameters["states"] == 2
    invalid = sg.skyrms_game("SK-SIG")
    invalid.fidelity["game"]["status"] = "explicit_fixture"
    for input_value in (profile, specification):
        with pytest.raises(ValueError, match="source-defined"):
            sg.build_adaptive(input_value, game=invalid)
        with pytest.raises(ValueError, match="source-defined"):
            constructor(input_value, game=invalid)
    variant = sg.skyrms_game("SK-SIG", {"state_probabilities": [0.2, 0.8]})
    simulation = constructor(specification, game=variant)
    assert (
        simulation.model_instance.specification.to_dict()["game"]["resolved_game"]
        == simulation.game.to_dict()
    )
    assert (
        sg.restore_adaptive(simulation.snapshot(), game=variant).snapshot()
        == simulation.snapshot()
    )
    with pytest.raises(ValueError, match="game override"):
        simulation.game = variant


def test_imported_template_state_and_provenance_restore_without_reopening_artifact(
    tmp_path, monkeypatch
):
    template = sg.build("nand.demo", 31)
    template.run(plays=10)
    artifact = template.checkpoint(tmp_path / "template")
    profile = sg.load_profile("tr-log.demo")
    profile.parameters.update(
        template_mode="artifact", templates={"template": str(artifact)}
    )
    transfer = sg.build(profile, 32)
    inspection = transfer.model_instance.inspect()
    assert inspection["construction_provenance"]["template"]["artifact_hash"]
    assert all(
        t.frozen
        for t in transfer.runtime.tables.values()
        if t.module.startswith("template:")
    )
    artifact.unlink()
    restored = sg.Simulation.restore(transfer.snapshot())
    assert restored.model_instance.inspect() == inspection
    assert transfer.step() == restored.step()
    assert transfer.snapshot() == restored.snapshot()


def test_finite_adapter_restores_embedded_template_state_without_external_artifacts(
    tmp_path,
):
    template = sg.build("nand.demo", 31)
    template.run(plays=10)
    artifact = template.checkpoint(tmp_path / "finite-template")
    transfer = sg.load_profile("tr-log.demo")
    transfer.parameters.update(
        template_mode="artifact", templates={"template": str(artifact)}
    )
    profile = sg.AdaptiveProfile(
        "finite-transfer",
        "TR-LOG",
        {"profile": transfer.to_dict()},
        method="reinforcement.conditional.R1",
    )
    simulation = sg.build(profile, 32)
    before = simulation.model_instance.inspect()
    artifact.unlink()
    restored = sg.restore_adaptive(simulation.snapshot())
    assert restored.model_instance.inspect() == before
    assert simulation.step() == restored.step()
    assert simulation.snapshot() == restored.snapshot()


def test_live_inventory_follows_failed_invention_and_structural_removal():
    simulation = sg.build("inv.demo", 2)
    model = simulation.model_instance
    address = simulation.runtime.add_table("R", "new", [0, 1], {"rule": "R1"})
    assert address in model.organisation.inventory(model.realisation)["R"]
    simulation.runtime.insert(address, 2)
    assert model.inspect()["realisation"]["tables"][address]["options"] == [0, 1, 2]
    simulation.runtime.remove(address)
    assert address not in model.organisation.inventory(model.realisation)["R"]


def test_outer_protocol_is_separate_from_model_conditions_and_controls_execution():
    specification = sg.ModelSpecification.from_profile("sig.iterative.demo")
    simulation = sg.build(specification, protocol={"plays": 4, "window": 2})
    simulation.run()
    assert simulation.runtime.clocks["play"] == 4
    assert simulation.metrics.recent.maxlen == 2
    assert simulation.model_instance.specification.to_dict() == specification.to_dict()
    assert "phase_length" in specification.realisation_parameters
    assert (
        sg.ModelSpecification.from_profile("meta-rule.demo").game_parameters[
            "inner_plays"
        ]
        == 10
    )
    assert sg.ModelSpecification.from_profile("change.demo").game_parameters[
        "change_at"
    ]
    assert (
        sg.Simulation.restore(simulation.snapshot()).snapshot() == simulation.snapshot()
    )


def test_game_override_retains_registered_and_native_source_questions():
    specification = sg.ModelSpecification.from_profile("sk-sig.r1.demo")
    declaration = specification.to_dict()["source"]
    declaration["identifiers"] = ["research:explicit-assay"]
    declaration["questions"] += ["NATIVE-Q"]
    specification = sg.ModelSpecification.resolve(
        game=specification.game_id,
        game_parameters=specification.game_parameters,
        realisation=specification.realisation_id,
        realisation_parameters=specification.realisation_parameters,
        organisation=specification.organisation_parameters,
        source=sg.SourceDeclaration(**declaration),
    )
    override = specification.construct_finite_game()
    override.fidelity["game"]["questions"].append("OVERRIDE-Q")
    simulation = sg.build_adaptive(specification, game=override)
    resolved = simulation.model_instance.specification.to_dict()["source"]
    assert {"NATIVE-Q", "OVERRIDE-Q"} <= set(resolved["questions"])
    assert "research:explicit-assay" in resolved["identifiers"]
    assert (
        sg.restore_adaptive(simulation.snapshot(), game=override).snapshot()
        == simulation.snapshot()
    )


@pytest.mark.parametrize(
    "profile", ["sk-sig.r1.demo", "sk-sig.strategy-r1.demo", "sk-sig.shared.demo"]
)
def test_inspection_addresses_actual_state_and_separate_decision_bindings(profile):
    simulation = sg.build(profile)
    inspection = simulation.model_instance.inspect()
    addresses = {
        address for group in inspection["state_addresses"].values() for address in group
    }
    realised_state = (
        simulation.frequencies
        if isinstance(simulation, sg.EvolutionSimulation)
        else simulation.weights
    )
    assert addresses == set(realised_state)
    assert inspection["decision_bindings"]["sender"]["state:0"] == "sender"


def test_direct_table_probability_query_preserves_empty_learner_support():
    table = sg.Table("module", "context", [0, 1], np.ones(2), {})
    np.testing.assert_array_equal(table.probabilities(), [0.5, 0.5])


def test_inventory_resolves_custom_slot_names_by_exact_binding():
    game = sg.FiniteGame(
        "custom-slots",
        ("a", "a/b"),
        (
            sg.FiniteDecision("a", "input", (0, 1), "first"),
            sg.FiniteDecision("a/b", "input", (0, 1), "second"),
        ),
        lambda provider: sg.ParticipantOutcome(False, (0, 0)),
        {},
        {},
    )
    organisation = sg.Organisation.for_finite(game)
    realisation = SimpleNamespace(
        id="finite.conditional",
        weights={"a/input": np.ones(2), "a/b/input": np.ones(2)},
    )
    assert organisation.inventory(realisation) == {
        "a": ["a/input"],
        "a/b": ["a/b/input"],
    }
