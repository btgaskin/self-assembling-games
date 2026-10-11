import copy

import numpy as np
import pytest

from self_assembling_games import (
    build,
    evaluate,
    load_profile,
    nand_transfer_closure,
    profiles,
)
from self_assembling_games.games import expected_signaling


@pytest.mark.parametrize("profile", profiles(), ids=lambda p: p["id"])
def test_every_registered_profile_runs_and_resumes(profile):
    from self_assembling_games import Simulation

    sim = build(profile, seed=2)
    sim.run(plays=12)
    resumed = Simulation.restore(sim.snapshot())
    sim.run(plays=7)
    resumed.run(plays=7)
    assert sim.snapshot() == resumed.snapshot()


@pytest.mark.parametrize(
    "identifier,count,weights",
    [
        ("sig.demo", 4, 8),
        ("cod.demo", 12, 32),
        ("nand.demo", 8, 16),
        ("hier.demo", 18, 50),
        ("scod.demo", 17, 51),
        ("ask.demo", 5, 13),
        ("dial.demo", 29, 158),
        ("dial-restricted.demo", 15, 58),
        ("att-r.demo", 9, 19),
    ],
)
def test_structural_counts(identifier, count, weights):
    sim = build(identifier)
    assert len(sim.runtime.tables) == count
    assert sum(len(t.weights) for t in sim.runtime.tables.values()) == weights


def test_exact_baselines_and_perfect_signaling():
    sim = build("sig.demo")
    assert evaluate(sim)["success_probability"] == pytest.approx(0.5)
    for x in [0, 1]:
        sim.set_policy("sender", x, x)
        sim.set_policy("receiver", x, x)
    assert evaluate(sim)["success_probability"] == 1
    nand = build("nand.demo")
    for context in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        nand.set_policy("R", context, 1)
    assert evaluate(nand)["success_probability"] == pytest.approx(0.75)


def test_coding_partition_and_hierarchical_efficiency():
    coding = build("cod.demo")
    hierarchy = build("hier.demo")
    for x in range(4):
        for sim in [coding, hierarchy]:
            sim.set_policy("A", x, x // 2)
            sim.set_policy("B", x, x % 2)
    for a in [0, 1]:
        for b in [0, 1]:
            for sim in [coding, hierarchy]:
                sim.set_policy("R", [a, b], 2 * a + b)
    for context, active in enumerate(["A", "B", "AB"]):
        hierarchy.set_policy("ES", context, active)
        hierarchy.set_policy("ER", active, context)
    assert evaluate(coding)["success_probability"] == 1
    evaluated = evaluate(hierarchy)
    assert evaluated["success_probability"] == pytest.approx(1)
    assert evaluated["expected_terms"] == pytest.approx(4 / 3)
    assert evaluated["expected_net_payoff"] == pytest.approx(1)


def test_hier_inactive_sender_and_receiver_observation():
    sim = build("hier.demo")
    for x in range(3):
        sim.set_policy("ES", x, "A")
    before = {
        k: copy.deepcopy(t.to_dict())
        for k, t in sim.runtime.tables.items()
        if t.module == "B"
    }
    sim.runtime.record = True
    sim.step()
    assert before == {
        k: t.to_dict() for k, t in sim.runtime.tables.items() if t.module == "B"
    }
    decisions = [e for e in sim.runtime.events if e["event"] == "decision"]
    receiver = next(e for e in decisions if e["module"] == "R")
    assert receiver["observation"] == {
        "expression": {"A": receiver["observation"]["expression"]["A"]}
    }
    assert any(e.get("label") == "missing_B" for e in sim.runtime.events)


def test_invention_success_and_failure_transaction():
    profile = load_profile("inv.demo")
    profile.parameters["n"] = 1
    sim = build(profile)
    sim.runtime.record = True
    sim.step()
    table = sim.runtime.tables["sender/0"]
    assert table.options == ["mutator", "sender:signal:0"]
    assert table.weights.tolist() == [1, 2]
    assert sim.runtime.tables['R/"sender:signal:0"'].weights.tolist() == [2]
    sim = build("inv.demo")
    sim.runtime.forced = [0, 1]
    sim.step()
    assert all(t.options == ["mutator"] for t in sim.runtime.tables.values())
    assert sim.runtime.next_identity == 1


def test_reciprocal_unsampled_host_update():
    profile = load_profile("vis.demo")
    profile.parameters.update(agents=2, schedule="ordered")
    sim = build(profile)
    sim.runtime.record = True
    sim.step()
    assert sim.runtime.tables['visit:0/"start"'].weights.tolist() == [2]
    assert sim.runtime.tables['visit:1/"start"'].weights.tolist() == [2]
    assert len([e for e in sim.runtime.events if e["event"] == "decision"]) == 1


def test_inquiry_self_bypasses_noise_and_reset_preserves_weights():
    profile = load_profile("inq.demo")
    profile.parameters.update(
        agents=1, epistemic=[1.0], social=[0.0], rounds_per_problem=2
    )
    sim = build(profile)
    sim.world["beliefs"] = [True]
    table = sim.runtime.tables['visit:0/"start"']
    table.current = 0
    sim.step()
    assert sim.world["beliefs"] == [True]
    assert table.weights.tolist() == [1, 2]
    sim.step()
    assert sim.world["beliefs"] == [False]
    assert table.weights.tolist() == [1, 3]
    assert sim.runtime.clocks["problem"] == 1


def test_inquiry_immediate_update_and_consulter_reliability():
    profile = load_profile("inq.demo")
    profile.parameters.update(
        agents=2, epistemic=[1.0, 0.0], social=[0.0, 1.0], rounds_per_problem=100
    )
    sim = build(profile)
    sim.interaction["inquiry_order"] = [0, 1]
    sim.runtime.tables['visit:0/"start"'].current = "nature"
    sim.runtime.tables['visit:1/"start"'].current = 0
    sim.step()
    sim.step()
    assert sim.world["beliefs"] == [True, True]


def test_composition_freezing_and_once_only_fanout():
    sim = build("comp-nand.2.demo")
    sim.set_policy("P", "start", "first")
    sim.set_policy("Q", "start", "first")
    before = {
        k: t.to_dict()
        for k, t in sim.runtime.tables.items()
        if t.module.startswith(("first:", "second:"))
    }
    sim.runtime.record = True
    sim.step()
    first = [
        e
        for e in sim.runtime.events
        if e["event"] == "decision" and e["module"].startswith("first:")
    ]
    assert len(first) == 3
    assert before == {
        k: t.to_dict()
        for k, t in sim.runtime.tables.items()
        if t.module.startswith(("first:", "second:"))
    }


def test_attention_binding_and_nand_transfer_closure():
    separated = build("att-b.source_value.demo")
    shared = build("att-b.value.demo")
    assert (
        len([t for t in separated.runtime.tables.values() if t.module == "sender"]) == 6
    )
    assert len([t for t in shared.runtime.tables.values() if t.module == "sender"]) == 2
    assert len(nand_transfer_closure()) == 10
    assert len(nand_transfer_closure(bijective=True)) == 4
    assert len(nand_transfer_closure(complement_output=True)) == 14
    assert (0, 1, 1, 0) not in nand_transfer_closure(complement_output=True)
    assert (1, 0, 0, 1) not in nand_transfer_closure(complement_output=True)


def test_ffsr_expected_payoffs_have_role_and_state_average():
    assert expected_signaling(5, 5) == 1.0
    assert expected_signaling(5, 10) == 0.0
    assert expected_signaling(0, 0) == 0.5


def test_meta_inner_reset_and_outer_persistence():
    sim = build("meta-rule.demo")
    before = sim.runtime.tables['choice/"start"'].weights.copy()
    sim.step()
    first = sim.interaction["last_inner"]
    sim.step()
    second = sim.interaction["last_inner"]
    assert first["seed"] != second["seed"]
    assert sim.runtime.clocks["problem"] == 2
    assert np.all(sim.runtime.tables['choice/"start"'].weights >= before)
    assert len(sim.runtime.tables) == 1


def test_evaluation_preserves_training_rng_and_state():
    sim = build("nand.demo")
    sim.run(plays=10)
    before = sim.snapshot()
    evaluate(sim)
    evaluate(sim, "sampled", samples=30, seed=99)
    assert sim.snapshot() == before
