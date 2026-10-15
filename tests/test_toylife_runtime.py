"""Native end-to-end acceptance: replay, pending work and detached assessment."""

import json

import numpy as np
import pytest

from self_assembling_games import model
from self_assembling_games.system import System


@pytest.fixture(scope="module", autouse=True)
def native_available():
    pytest.importorskip("toylife2")
    from toylife2.tables import DEFAULT_SOURCE

    if not DEFAULT_SOURCE.is_dir():
        pytest.skip("canonical toyLIFE source tables are unavailable")


def small(name="boolean.nand", *, quantum=None, initial=None, task=None):
    parameters = {
        "backend": "cpu",
        "response_updates": 2,
        "episodes": 1,
        "trace_limit": 256,
    }
    if quantum is not None:
        parameters["max_chemical_updates"] = quantum
    overrides = {
        "adaptation": {"parameters": {"population_size": 2}},
        "execution": {"budget": 2, "parameters": parameters},
    }
    if initial is not None:
        overrides["realisation"] = {"parameters": {"initial_genotypes": initial}}
    if task:
        overrides["interaction"] = {"parameters": task}
    return model("toylife." + name, **overrides)


def finish(system, position, max_calls=256):
    for _ in range(max_calls):
        if system.position == position and system.inspect()["pending"] is None:
            return system
        system.advance(position - system.position)
    pytest.fail("bounded work did not reach requested generation")


def test_native_evolution_checkpoint_and_detached_evaluation():
    declaration = small()
    full = declaration.instantiate(3).advance(2)
    split = declaration.instantiate(3).advance(1)
    restored = System.restore(json.loads(json.dumps(split.snapshot())))
    restored.advance(1)
    assert restored.position == full.position == 2
    assert restored.inspect()["genotypes"] == full.inspect()["genotypes"]
    assert restored.evaluate()["scores"] == full.evaluate()["scores"]
    before = restored.snapshot()
    detached = restored.evaluate()
    assert detached["status"] == "complete"
    assert restored.snapshot() == before
    inspected = restored.inspect()
    inspected["apparatus"]["dose"] = 99
    assert restored.snapshot() == before


@pytest.mark.parametrize(
    "name,task",
    [("delay", {"cue_updates": 2, "delay": 1}), ("query", {"max_queries": 1})],
)
def test_episodic_pending_quantum_restore_matches_uninterrupted(name, task):
    uninterrupted = small(name, task=task).instantiate(3).advance(1)
    limited = small(name, quantum=1, task=task).instantiate(3)
    limited.advance(1)
    assert limited.position == 0
    assert limited.inspect()["pending"] is not None
    assert limited.inspect()["work"]["chemical_updates"] == 1
    restored = System.restore(json.loads(json.dumps(limited.snapshot())))
    finish(restored, 1)
    assert restored.inspect()["genotypes"] == uninterrupted.inspect()["genotypes"]
    assert restored.evaluate()["scores"] == uninterrupted.evaluate()["scores"]
    assert restored.inspect()["trace"] == uninterrupted.inspect()["trace"]
    assert restored.inspect()["work"] == uninterrupted.inspect()["work"]


def test_initial_population_and_disjoint_ordering_assessment():
    initial = ["0" * 20 + "1" * 20, "0" * 19 + "1" + "1" * 19 + "0"]
    system = small(
        "ordering", initial=initial, task={"size": 3, "train_pairs": "adjacent"}
    ).instantiate(3)
    from toylife2.genotypes import encode

    np.testing.assert_array_equal(system.inspect()["genotypes"][0], encode(initial))
    before = system.snapshot()
    train = system.evaluate()
    held = system.evaluate(split="held_out")
    assert train["status"] == held["status"] == "complete"
    assert held["split"] == "held_out"
    assert system.snapshot() == before


@pytest.mark.parametrize("name", ["boolean.nand", "query"])
def test_candidate_batch_order_invariance_and_bounded_analysis(name):
    from self_assembling_games.toylife.runtime import AssayEvaluator

    declaration = small(name, task={"max_queries": 1} if name == "query" else None)
    system = declaration.instantiate(3)
    genes = np.array(system.inspect()["genotypes"][0], dtype=np.int32)
    native = system._backend.model
    evaluator = AssayEvaluator(native, declaration.to_dict(), 3)
    combined = evaluator.evaluate(genes, panel=7)
    reversed_result = evaluator.evaluate(genes[::-1].copy(), panel=7)
    np.testing.assert_array_equal(combined.scores, reversed_result.scores[::-1])
    for index, genome in enumerate(genes):
        single = evaluator.evaluate(genome[None], panel=7)
        assert single.scores[0] == combined.scores[index]
        assert len(evaluator.requests) == 1
    report = system.evaluate(panel=7)
    assert report["selection_denominator"] == 1 << 24
    assert len(report["selection_weights"]) == len(genes)
    if name == "query":
        curves = report["episodic_trial_curves"]
        assert len(curves["candidate_episode_scores"]) == len(genes)
        assert curves["mean_by_scored_trial"]
    else:
        assert len(report["case_metrics"]) == 4
        confusion = np.array(report["confusion"]["target_by_action_probability"])
        assert confusion.sum() == pytest.approx(1)
        assert np.trace(confusion) == pytest.approx(report["success_probability"])


def test_phase_truncation_reports_status_and_prevents_selection():
    declaration = small("delay", task={"cue_updates": 2, "delay": 1})
    data = declaration.to_dict()
    data["execution"]["parameters"]["max_episode_phases"] = 1
    system = model(data).instantiate(3)
    before = system.inspect()["genotypes"]
    report = system.evaluate()
    assert report["status"] == "truncated"
    assert report["candidate_statuses"] == ["truncated", "truncated"]
    assert report["status_detail"] == "task_phase_limit"
    assert "selection_weights" not in report
    system.advance(1)
    assert system.position == 0
    assert system.inspect()["genotypes"] == before


def test_ordering_full_training_reports_held_out_overlap():
    system = small("ordering", task={"size": 3, "train_pairs": "all"}).instantiate(3)
    assert system.evaluate(split="held_out")["evaluation_overlap"] is True


def test_analysis_allocation_is_admitted_before_request_creation():
    from self_assembling_games.toylife.runtime import AssayEvaluator

    declaration = small("query", task={"max_queries": 1})
    system = declaration.instantiate(3)
    data = declaration.to_dict()
    data["execution"]["parameters"]["memory_budget_mib"] = 0
    evaluator = AssayEvaluator(system._backend.model, data, 3)
    genes = np.array(system.inspect()["genotypes"][0], dtype=np.int32)
    with pytest.raises(MemoryError, match="analysis"):
        evaluator.evaluate(genes)
    assert evaluator.requests == {}


def test_phase_reset_matches_fresh_native_cells_and_quantum_replay(monkeypatch):
    from toylife2.chemistry import ChemistrySession
    from self_assembling_games.toylife.apparatus import controller, encode

    declaration = small("delay", task={"cue_updates": 1, "delay": 1})
    data = declaration.to_dict()
    realisation = data["realisation"]["parameters"]
    realisation.update(initial_genotypes=["0" * 40], reset_between_phases=True)
    realisation["controller"]["initial_proteins"] = [[47, 1]]
    realisation["apparatus"]["ligands"] = {"act": "1111"}
    reset_model = model(data)
    reset = reset_model.instantiate(3)
    constructions = []
    original_init = ChemistrySession.__init__

    def counted_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructions.append(self)

    monkeypatch.setattr(ChemistrySession, "__init__", counted_init)
    reset.advance(1)
    trace = reset.inspect()["trace"]
    assert len(constructions) == len(trace)
    monkeypatch.setattr(ChemistrySession, "__init__", original_init)
    apparatus = reset.inspect()["apparatus"]
    spec = controller(apparatus, realisation["controller"])
    for row in trace:
        fresh = ChemistrySession(
            reset._backend.model, np.array(row["genotype"], np.int32), spec
        )
        # Cue, rest, zero-duration decision and feedback are distinct phase baselines.
        phase = row["phase"]
        duration = 0 if phase == 2 else 1
        for _ in range(duration):
            fresh.step(spec.interface.encode(encode(apparatus, row["observation"])))
        for name, counts in fresh.last_readouts.items():
            np.testing.assert_array_equal(counts, row["readouts"][name])
    retained_data = reset_model.to_dict()
    retained_data["realisation"]["parameters"]["reset_between_phases"] = False
    retained = model(retained_data).instantiate(3).advance(1)
    assert retained.inspect()["work"] == reset.inspect()["work"]
    assert any(
        a["readouts"] != b["readouts"]
        for a, b in zip(trace, retained.inspect()["trace"], strict=True)
    )
    limited_data = reset_model.to_dict()
    limited_data["execution"]["parameters"]["max_chemical_updates"] = 1
    limited = model(limited_data).instantiate(3).advance(1)
    restored = System.restore(json.loads(json.dumps(limited.snapshot())))
    finish(restored, 1)
    assert restored.inspect()["trace"] == trace
    assert restored.inspect()["work"] == reset.inspect()["work"]
    assert restored.inspect()["genotypes"] == reset.inspect()["genotypes"]


def test_frozen_evaluation_can_use_fresh_panels_and_more_episodes():
    system = small("delay", task={"cue_updates": 1, "delay": 1}).instantiate(3)
    before = system.snapshot()
    report = system.evaluate(panel=1000000, episodes=3)
    assert report["panel"] == 1000000
    assert report["sampling"]["episodes"] == 3
    assert report["sampling"]["independent_population_lifetimes"] is False
    assert system.snapshot() == before
    assert report == system.evaluate(panel=1000000, episodes=3)
    for kwargs in ({"episodes": 0}, {"episodes": True}, {"panel": -1}, {"panel": True}):
        with pytest.raises(ValueError):
            system.evaluate(**kwargs)
    with pytest.raises(ValueError, match="episodic"):
        small().instantiate(3).evaluate(episodes=2)
