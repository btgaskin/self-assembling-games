import copy
import json

import pytest

from self_assembling_games import (
    Profile,
    build,
    catalogue,
    evaluate,
    export_template,
    held_out_ordering,
    load_profile,
    replay_trace,
    resume,
    run_ensemble,
)
from self_assembling_games.cli import main


@pytest.mark.parametrize("identifier", ["sig.strategy.demo", "inv.forget-type.demo"])
def test_disk_checkpoint_preserves_decision_order(identifier, tmp_path):
    profile = load_profile(identifier)
    profile.parameters["n"] = 12
    original = build(profile, seed=71)
    original.run(plays=60)
    restored = resume(original.checkpoint(tmp_path / "checkpoint"))
    assert list(restored.runtime.tables) == list(original.runtime.tables)
    original.run(plays=100)
    restored.run(plays=100)
    assert original.snapshot() == restored.snapshot()


@pytest.mark.parametrize(
    "identifier",
    [
        "sig.iterative.demo",
        "hier.demo",
        "inv.demo",
        "ord.demo",
        "dial.demo",
        "inq.demo",
        "meta-rule.demo",
        "ffsr.demo",
    ],
)
def test_full_event_replay(identifier, tmp_path):
    sim = build(identifier, seed=5)
    log = tmp_path / "trace.jsonl"
    sim.run(plays=15, trace=log)
    replayed = replay_trace(log)
    assert replayed.snapshot() == sim.snapshot()


def test_append_replay_and_checkpoint_resume(tmp_path):
    sim = build("nand.demo", seed=4)
    trace = tmp_path / "trace.jsonl"
    sim.run(plays=10, trace=trace, output=tmp_path / "run")
    resumed = resume(tmp_path / "run")
    resumed.run(plays=5, trace=trace)
    assert replay_trace(trace).snapshot() == resumed.snapshot()


def test_learned_nand_artifact_remains_stochastic_and_frozen(tmp_path):
    original = build("nand.demo", seed=7)
    original.run(plays=30)
    checkpoint = original.checkpoint(tmp_path / "nand")
    p = load_profile("tr-log.demo")
    p.parameters.update(
        template_mode="artifact", templates={"template": str(checkpoint)}
    )
    transfer = build(p)
    frozen = {
        k: copy.deepcopy(t.to_dict())
        for k, t in transfer.runtime.tables.items()
        if t.frozen
    }
    assert all(t.current is None for t in transfer.runtime.tables.values() if t.frozen)
    transfer.run(plays=20)
    assert frozen == {
        k: t.to_dict() for k, t in transfer.runtime.tables.items() if t.frozen
    }
    assert transfer.lineage["template"]["play"] == 30


def test_vocabulary_export_then_learned_order_transfer(tmp_path):
    original = build("ord.demo", seed=3)
    original.run(plays=100)
    before = original.snapshot()
    checkpoint = export_template(
        original, tmp_path / "ordering", suppress_mutator=True, complete_pair_rows=True
    )
    assert original.snapshot() == before
    p = load_profile("tr-ord.demo")
    p.parameters.update(
        template_mode="artifact", templates={"ordering": str(checkpoint)}
    )
    transfer = build(p)
    transfer.run(plays=20)
    assert all(
        "mutator" not in t.options for t in transfer.runtime.tables.values() if t.frozen
    )
    assert held_out_ordering(transfer, samples=20)["samples"] == 20


def test_shared_stochastic_ancestor_exact_evaluation():
    sim = build("comp-nand.2.demo")
    sim.set_policy("P", "start", "first")
    sim.set_policy("Q", "start", "first")
    for table in sim.runtime.tables.values():
        if table.module == "first:R":
            table.current = None
    assert evaluate(sim)["success_probability"] == pytest.approx(0.5)


def test_profile_gates_and_unsupported_learner():
    p = load_profile("att-b.value.demo")
    p.kind = "book_literal"
    with pytest.raises(ValueError, match="Q8"):
        p.validate()
    p = load_profile("ord.demo")
    del p.parameters["failed_row"]
    with pytest.raises(ValueError, match="failed_row"):
        p.validate()
    p = load_profile("hier.demo")
    p.learner = {"rule": "R1"}
    with pytest.raises(ValueError, match="unsupported"):
        p.validate()
    with pytest.raises(ValueError, match="awaiting"):
        build(Profile("ritualization", "RIT"))
    assert {"SIG", "NAND", "HIER", "ORD"} <= {entry["id"] for entry in catalogue()}


def test_resource_limits_and_censoring_are_measured():
    p = load_profile("inv.demo")
    p.parameters["max_tables"] = 3
    sim = build(p)
    sim.run(plays=100)
    assert sim.status == "resource_limit"
    assert sim.metrics.completed == 0 and sim.metrics.attempts == 1
    sim = build("ask.demo")
    sim.set_policy("assistant", "start", "Q")
    sim.set_policy("reply1", 0, "Q")
    sim.set_policy("reply1", 1, "Q")
    sim.set_policy("builder", ["Q", 0], 0)
    sim.set_policy("builder", ["Q", 1], 0)
    result = evaluate(sim, max_branches=200000)
    assert result["completed_mass"] == 0
    assert result["truncated_mass"] == pytest.approx(1)


def test_ensemble_is_resumable_and_protocol_checked(tmp_path):
    first = run_ensemble("sig.demo", [1, 2], output=tmp_path, plays=20, threshold=0.5)
    second = run_ensemble("sig.demo", [1, 2], output=tmp_path, plays=20, threshold=0.5)
    assert first == second
    assert len(first["threshold_wilson_95"]) == 2
    with pytest.raises(ValueError, match="different protocol"):
        run_ensemble("sig.demo", [1, 2], output=tmp_path, plays=21)


def test_cli_end_to_end(tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["run", "sig.demo", "--plays", "10", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["evaluate", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "exact"
    assert main(["resume", str(output), "--plays", "5"]) == 0
    capsys.readouterr()
    assert resume(output).runtime.clocks["play"] == 15
    assert main(["run", "sig.demo", "--output", str(output)]) == 2
    assert "resume" in capsys.readouterr().err


def test_inspection_copies_options_and_learner_state():
    sim = build("sig.demo")
    view = sim.inspect()
    view["tables"]["sender/0"]["options"].append("changed")
    view["tables"]["sender/0"]["learner"]["rule"] = "changed"
    assert sim.runtime.tables["sender/0"].options == [0, 1]
    assert sim.runtime.tables["sender/0"].learner["rule"] == "R1"
