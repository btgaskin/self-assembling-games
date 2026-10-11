import pytest

from self_assembling_games import (
    Profile,
    build,
    conventional_kinds,
    evaluate,
    load_profile,
    run_ensemble,
)


def test_semantic_baselines_for_ordering_addition_and_box():
    # The exact finite source distributions, independently of adaptive trajectories.
    pairs = [(a, b) for a in range(7) for b in range(7)]
    assert sum(a < b for a, b in pairs) / len(pairs) == pytest.approx(3 / 7)
    assert sum(a == b for a, b in pairs) / len(pairs) == pytest.approx(1 / 7)
    assert sum(a + b == 5 for a in range(6) for b in range(6)) / 36 == pytest.approx(
        1 / 6
    )
    assert (0.5 * 10 + 0.5 * 9) / 10 == 0.95
    assert (2 / 3) ** 2 + (1 / 3) ** 2 == pytest.approx(5 / 9)


def test_nonadjacent_evaluation_keeps_training_state():
    from self_assembling_games import held_out_ordering

    sim = build("tr-part.demo")
    before = sim.snapshot()
    held_out_ordering(sim, samples=50)
    assert sim.snapshot() == before


def test_forgetting_extinction_and_mutator_protection():
    from self_assembling_games.games import forget_signal

    sim = build("inv.forget-type.demo")
    rt = sim.runtime
    identity = rt.table_id("sender", 0)
    rt.insert(identity, "rare")
    receiver = rt.add_table("R", "rare", [0, 1, 2], {"rule": "R1"})
    rt.forced = [0]
    forget_signal(rt, sim.profile.parameters["forgetting"])
    assert rt.tables[identity].options == ["mutator"] and receiver not in rt.tables
    assert rt.tables[identity].weights.tolist() == [1]


def test_failed_existing_pair_creates_a_persistent_row():
    sim = build("ord.demo")
    rt = sim.runtime
    for module in ["A", "B"]:
        for x in range(7):
            identity = rt.table_id(module, x)
            rt.insert(identity, module + "0")
            rt.tables[identity].current = module + "0"
    # nature 0,0; receiver chooses '<', a wrong act for equality.
    rt.forced = [0, 0, 0]
    sim.step()
    assert rt.table_id("R", ["A0", "B0"]) in rt.tables


def test_conventional_kinds_use_actual_policy():
    sim = build("cod.demo")
    for x in range(4):
        sim.set_policy("A", x, x // 2)
    assert conventional_kinds(sim, "A") == [[0, 1], [2, 3]]


def test_validation_rejects_ambiguous_or_invalid_environment_fields():
    cases = [
        ("att-b.value.demo", "binding", "hidden"),
        ("box.demo", "placement", "sometimes"),
        ("tr-part.demo", "pair_sampler", "random"),
        ("comp-nand.1.demo", "selectors", 3),
        ("mh.demo", "host", "uninformed"),
        ("vis.demo", "agents", 1),
        ("meta-oracle.demo", "answer_process", "correlated"),
    ]
    for identifier, field, value in cases:
        profile = load_profile(identifier)
        profile.parameters[field] = value
        with pytest.raises(ValueError):
            profile.validate()


def test_exact_evaluation_branch_limit_and_rng_isolation():
    sim = build("hier.demo")
    before = sim.snapshot()
    with pytest.raises(ValueError, match="max_branches"):
        evaluate(sim, max_branches=2)
    assert sim.snapshot() == before


def test_parallel_ensembles_preserve_seed_results(tmp_path):
    serial = run_ensemble(
        "sig.demo", [4, 7], output=tmp_path / "serial", plays=30, workers=1
    )
    parallel = run_ensemble(
        "sig.demo", [4, 7], output=tmp_path / "parallel", plays=30, workers=2
    )
    assert serial == parallel


def test_success_matrix_can_reward_a_successful_act_negatively():
    profile = Profile(
        "signed-match",
        "SIG",
        {
            "states": 1,
            "signals": 1,
            "actions": 1,
            "payoffs": [[-0.5]],
            "success_matrix": [[True]],
        },
        {"rule": "SIGNED"},
    )
    sim = build(profile)
    outcome = sim.step()
    assert outcome.success and outcome.payoff == -0.5
    assert sim.runtime.tables["sender/0"].weights.tolist() == [1]


def test_existing_resource_limit_is_stable_on_ensemble_resume(tmp_path):
    profile = load_profile("inv.demo")
    profile.parameters["max_tables"] = 3
    first = run_ensemble(profile, [0], output=tmp_path, plays=20)
    assert run_ensemble(profile, [0], output=tmp_path, plays=20) == first
