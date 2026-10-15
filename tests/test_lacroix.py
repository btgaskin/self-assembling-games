import copy

import numpy as np
import pytest

from self_assembling_games import System, build, evaluate, load_profile, model, replay_trace


PROFILES = [
    "lacroix.logic.stipulated",
    "lacroix.logic.role-free",
    "lacroix.logic.observation-control",
]


def fixture(sim, *, swapped=False):
    """An explicit fully communicating validation fixture, never source training."""
    full = sim.model_instance.parameters["sender_observation"] == "full_world"
    for m in ["A", "B"]:
        for table in [t for t in sim.runtime.tables.values() if t.module == m]:
            game = table.context // 2 if full else table.context
            input_value = table.context % 2 if full else table.context
            option = game if (m == "A") != swapped else input_value
            sim.set_policy(m, table.context, option, frozen=False)
    routes = ["B", "A", "AB"] if swapped else ["A", "B", "AB"]
    for c, active in enumerate(routes):
        sim.set_policy("ES", c, active, frozen=False)
        sim.set_policy("ER", active, c, frozen=False)
    for a in [0, 1]:
        for b in [0, 1]:
            sim.set_policy("R", [a, b], 2 * b + a if swapped else 2 * a + b, frozen=False)


@pytest.mark.parametrize("preset", PROFILES)
def test_source_domains_and_initialization(preset):
    system = model(preset).instantiate(seed=9)
    declaration = system.model.to_dict()
    assert "lacroix2022" in declaration["source"]["identifiers"]
    assert "https://doi.org/10.1093/bjps/axz049" in declaration["source"]["identifiers"]
    rt = system._backend.simulation.runtime
    count = 2 if preset.endswith("stipulated") else 4
    assert [t.context for t in rt.tables.values() if t.module == "A"] == list(range(count))
    assert [t.context for t in rt.tables.values() if t.module == "B"] == list(range(count))
    assert len(set(rt.owners.values())) == 5
    assert all(np.array_equal(t.weights, np.ones(len(t.options))) for t in rt.tables.values())
    assert all(not t.frozen and t.current is None for t in rt.tables.values())
    assert declaration["interaction"]["parameters"]["sender_observation"] == (
        "assigned_property" if count == 2 else "full_world"
    )


@pytest.mark.parametrize("preset", PROFILES)
def test_ideal_encoding_exact_evaluation_and_public_restore(preset):
    system = model(preset).instantiate(seed=5)
    fixture(system._backend.simulation)
    before = system.snapshot()
    assert system.evaluate()["success_probability"] == pytest.approx(1)
    assert system.snapshot() == before
    system.advance(15)
    restored = System.restore(system.snapshot())
    system.advance(12)
    restored.advance(12)
    assert restored.snapshot() == system.snapshot()


@pytest.mark.parametrize("context,allowed", [(0, {2, 3}), (1, {0, 2}), (2, {2})])
def test_figure8_contextual_correctness_is_symbolic(context, allowed):
    # Figure 8 fixes nature at (neg,0). Its three rows enumerate these actions.
    for decoded in range(4):
        sim = build("lacroix.logic.stipulated", seed=3)
        fixture(sim)
        for a in [0, 1]:
            for b in [0, 1]:
                sim.set_policy("R", [a, b], decoded, frozen=False)
        sim.runtime.forced = [1, 0, context, 0]
        outcome = sim.step()
        assert outcome.success == (decoded in allowed)
        if context == 2 and decoded == 0:
            assert outcome.details["boolean_correct"]
            assert not outcome.details["context_correct"]


def test_role_free_learns_swapped_sender_roles_but_control_retains_relevance():
    for preset, expected in [
        ("lacroix.logic.role-free", 1),
        ("lacroix.logic.observation-control", 1 / 3),
    ]:
        sim = build(preset)
        fixture(sim, swapped=True)
        assert evaluate(sim)["success_probability"] == pytest.approx(expected)


def test_involved_agents_only_receive_success_credit_and_missing_bit_is_uniform():
    sim = build("lacroix.logic.stipulated")
    fixture(sim)
    before = {k: t.weights.copy() for k, t in sim.runtime.tables.items()}
    sim.runtime.record = True
    sim.runtime.forced = [1, 0, 0, 1]  # neg, input0, game-context, completed missing B=1
    assert sim.step().success
    feedback = [e for e in sim.runtime.events if e["event"] == "feedback"]
    assert len(feedback) == 4
    assert {e["table"].split("/")[0] for e in feedback} == {"ES", "A", "R", "ER"}
    assert all(e["payoff"] == 1 and e["applied"] for e in feedback)
    assert next(e for e in sim.runtime.events if e.get("label") == "missing_B")["probabilities"] == [0.5, 0.5]
    receiver = next(e for e in sim.runtime.events if e["event"] == "decision" and e["module"] == "R")
    assert receiver["table"] == "R/[1,1]"
    assert receiver["observation"] == {"signals": {"A": 1}}
    for k, t in sim.runtime.tables.items():
        assert t.weights.sum() - before[k].sum() == (1 if k in {e["table"] for e in feedback} else 0)


def test_failed_relevance_has_zero_reinforcement_even_with_correct_tuple():
    sim = build("lacroix.logic.stipulated")
    fixture(sim)
    sim.set_policy("ES", 0, "AB", frozen=False)
    sim.set_policy("ER", "AB", 0, frozen=False)
    before = copy.deepcopy(sim.snapshot()["runtime"]["tables"])
    sim.runtime.forced = [1, 0, 0]
    outcome = sim.step()
    assert outcome.details["context_correct"]
    assert not outcome.details["relevance_correct"]
    assert not outcome.success
    assert outcome.payoff == 0
    assert sim.snapshot()["runtime"]["tables"] == before


def test_both_context_credits_all_five_agents():
    sim = build("lacroix.logic.stipulated")
    fixture(sim)
    before = {k: t.weights.sum() for k, t in sim.runtime.tables.items()}
    sim.runtime.forced = [1, 0, 2]
    assert sim.step().success
    changed = {t.module for k, t in sim.runtime.tables.items() if t.weights.sum() > before[k]}
    assert changed == {"A", "B", "ES", "R", "ER"}
    assert sum(t.weights.sum() - before[k] for k, t in sim.runtime.tables.items()) == 5


@pytest.mark.parametrize("preset", PROFILES)
def test_joint_training_full_trace_replay(preset, tmp_path):
    sim = build(preset, seed=14)
    before = {k: t.weights.copy() for k, t in sim.runtime.tables.items()}
    trace = tmp_path / "trace.jsonl"
    sim.run(plays=100, trace=trace)
    assert replay_trace(trace).snapshot() == sim.snapshot()
    assert all(not t.frozen for t in sim.runtime.tables.values())
    assert any(not np.array_equal(before[k], t.weights) for k, t in sim.runtime.tables.items())


def test_source_and_constructed_cases_cannot_be_silently_interchanged():
    p = load_profile("lacroix.logic.observation-control")
    p.kind = "source_resolved"
    with pytest.raises(ValueError, match="deliberate variant"):
        build(p)
    p = load_profile("lacroix.logic.stipulated")
    p.parameters["sender_observation"] = "full_world"
    with pytest.raises(ValueError, match="match the declared"):
        build(p)
    p = load_profile("lacroix.logic.stipulated")
    p.parameters["term_cost"] = 0.1
    with pytest.raises(ValueError, match="Unknown LaCroix"):
        build(p)
    p = load_profile("lacroix.logic.stipulated")
    p.learner["error"] = 0.01
    with pytest.raises(ValueError, match="without exploration"):
        build(p)
