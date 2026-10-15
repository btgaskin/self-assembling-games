"""Declared acquisition, state carry and recovery on retained phase boundaries."""

import json
import hashlib
from pathlib import Path
import shutil
import tomllib

import pytest

from self_assembling_games.config import dumps, from_dict
from self_assembling_games.domain import experiment, model
from self_assembling_games.experiments import Run, resume, run
from self_assembling_games.phase_experiments import phase_seed
from self_assembling_games.system import System
from self_assembling_games.simulation import digest


def read(path):
    return json.loads(Path(path).read_text())


def phases(**changes):
    declaration = {
        "schema_version": 3, "preset": "sig.demo", "seeds": [17],
        "phases": [
            {"id": "acquire", "budget": 3},
            {"id": "branch", "budget": 2, "start": {"kind": "carry", "phase": "acquire", "transition": {"kind": "identity"}}},
        ],
    }
    declaration.update(changes)
    return from_dict(declaration)


def phase_result(study, index=0, trial=0):
    return read(study.path / "results.json")["trials"][trial]["phases"][index]


def checkpoint(study, index=0, trial=0):
    return read(phase_result(study, index, trial)["checkpoint"])


def test_compact_phases_resolve_and_roundtrip_without_constructing(monkeypatch):
    monkeypatch.setattr(model("sig.demo").__class__, "instantiate", lambda *a, **k: pytest.fail("construction while resolving"))
    declaration = phases()
    resolved = declaration.to_dict()
    assert resolved["schema_version"] == 3
    assert resolved["conditions"][0]["phases"][1]["start"]["phase"] == "acquire"
    assert from_dict(tomllib.loads(dumps(declaration))) == declaration
    assert experiment("sig.demo", phases=[{"id": "one", "budget": 0}]).to_dict()["schema_version"] == 3


@pytest.mark.parametrize("change", [
    {"phases": [{"id": "a", "budget": 1, "start": {"kind": "carry", "phase": "later"}}]},
    {"phases": [{"id": "a"}, {"id": "a"}]},
    {"phases": [{"id": "a", "start": {"kind": "compose", "bindings": {"template": {"phase": "a"}}}}]},
    {"phases": [{"id": "a", "interventions": [{"kind": "trainability", "modules": [], "enabled": False}]}]},
    {"phases": [{"id": "a", "budget": 1.5}]},
])
def test_invalid_phase_declarations_are_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        phases(**change)


def test_carry_branches_restore_acquired_state_and_keep_parent_immutable(tmp_path):
    declaration = phases(phases=[
        {"id": "acquire", "budget": 3},
        {"id": "left", "budget": 2, "start": {"kind": "carry", "phase": "acquire"}},
        {"id": "right", "budget": 2, "start": {"kind": "carry", "phase": "acquire"}},
    ])
    study = run(declaration, directory=tmp_path)
    assert study.status()["status"] == "complete"
    source, left, right = (checkpoint(study, i) for i in range(3))
    expected = model("sig.demo").instantiate(seed=phase_seed(17, "acquire")).advance(3)
    assert source["state"] == expected.snapshot()["state"]
    assert left["state"] == right["state"]
    assert left["state"]["runtime"]["clocks"]["play"] == 5
    assert source["state"]["runtime"]["clocks"]["play"] == 3
    original = Path(phase_result(study)["checkpoint"]).read_bytes()
    study.evaluate()
    resume(study.path)
    assert Path(phase_result(study)["checkpoint"]).read_bytes() == original
    assert study.verify()["valid"]


def test_release_fixed_side_is_a_declared_descendant(tmp_path):
    study = run(phases(phases=[
        {"id": "cue", "preset": "cue.demo", "budget": 2},
        {"id": "mutual", "preset": "sig.demo", "budget": 0, "start": {
            "kind": "carry", "phase": "cue", "transition": {
                "kind": "release_fixed", "fixed_weight": 9.0, "other_weight": 1.0,
            },
        }},
    ]), directory=tmp_path)
    assert study.status()["status"] == "complete"
    source, target = checkpoint(study), checkpoint(study, 1)
    assert target["model"]["interaction"]["game"] == "SIG"
    for key, table in source["state"]["runtime"]["tables"].items():
        after = target["state"]["runtime"]["tables"][key]
        if table["module"] == "sender":
            assert table["frozen"] and after["current"] is None
            assert not after["frozen"]
            assert sorted(after["weights"]) == [1.0, 9.0]
        else:
            assert after == table
    assert study.verify()["valid"]


def test_phase_measurement_failure_retries_at_retained_boundary(tmp_path, monkeypatch):
    original = System.evaluate
    observed = []
    failed = False

    def evaluate(self, **kwargs):
        nonlocal failed
        observed.append(self.position)
        if self.position == 2 and not failed:
            failed = True
            raise ValueError("temporary measurement failure")
        return original(self, **kwargs)

    monkeypatch.setattr(System, "evaluate", evaluate)
    study = run(phases(phases=[{"id": "a", "budget": 5, "analysis": {"sample_interval": 2}}]), directory=tmp_path)
    assert study.status()["status"] == "complete_with_failures"
    assert study.verify()["valid"]
    assert read(study.path / "results.json")["summary"]["default"]["a"]["evaluated"] == 0
    resume(study.path)
    result = phase_result(study)
    assert [s["phase_position"] for s in result["samples"]] == [0, 2, 4, 5]
    assert observed[:3] == [0, 2, 2]
    assert study.status()["status"] == "complete"


def test_final_phase_extension_retains_old_endpoint_and_crossing(tmp_path, monkeypatch):
    original = System.evaluate

    def evaluate(self, **kwargs):
        value = original(self, **kwargs)
        value["success_probability"] = float(self.position == 5)
        return value

    monkeypatch.setattr(System, "evaluate", evaluate)
    study = run(phases(phases=[
        {"id": "ancestor", "budget": 0},
        {"id": "last", "budget": 5, "analysis": {"sample_interval": 2, "threshold": 0.9}},
    ]), directory=tmp_path)
    ancestor_path = Path(phase_result(study)["checkpoint"])
    ancestor = ancestor_path.read_bytes()
    endpoint_path = Path(phase_result(study, 1)["checkpoint"])
    endpoint = endpoint_path.read_bytes()
    assert phase_result(study, 1)["threshold"]["first_observed_position"] == 5
    resume(study.path, additional_budget=2)
    result = phase_result(study, 1)
    assert result["budget"] == 7 and result["position"] == 7
    assert result["threshold"]["reached"]
    assert result["threshold"]["first_observed_position"] == 5
    assert [s["phase_position"] for s in result["samples"]] == [0, 2, 4, 5, 6, 7]
    assert ancestor_path.read_bytes() == ancestor
    assert endpoint_path.read_bytes() == endpoint
    assert study.status()["resumptions"][-1]["scope"] == "final_phase"
    assert study.verify()["valid"]


def test_stale_manifest_cursor_and_partial_generation_recover(tmp_path):
    study = run(phases(phases=[{"id": "a", "budget": 3}]), directory=tmp_path)
    result = phase_result(study)
    phase_directory = Path(result["checkpoint"]).parents[2]
    (phase_directory / "cursor.json").write_text('{"sequence": 0}')
    partial = phase_directory / "commits" / "99999999"
    partial.mkdir()
    (partial / "checkpoint.json").write_text("incomplete")
    manifest = read(study.path / "manifest.json")
    manifest["status"] = "running"
    (study.path / "manifest.json").write_text(json.dumps(manifest))
    assert study.verify()["valid"]
    before = checkpoint(study)
    resume(study.path)
    assert checkpoint(study) == before
    assert study.status()["status"] == "complete"


@pytest.mark.parametrize("phased", [False, True])
def test_hard_stop_after_receipt_before_cursor_recovers_boundary(tmp_path, monkeypatch, phased):
    from self_assembling_games import experiments as legacy
    from self_assembling_games import phase_experiments as staged

    module = staged if phased else legacy
    original = module._write
    stopped = False

    def interrupt(path, value):
        nonlocal stopped
        path = Path(path)
        boundary = path.name == "cursor.json" if phased else (
            path.name == "checkpoint.json" and path.parent.name.startswith("seed-")
            and value["state"]["runtime"]["clocks"]["play"] == 2
        )
        if phased and boundary:
            receipt = read(path.parent / "commits" / f"{value['sequence']:08d}" / "receipt.json")
            boundary = receipt["position"] == 2 and receipt["pending_measurement"]
        if boundary and not stopped:
            stopped = True
            raise KeyboardInterrupt("simulated hard stop after receipt commit")
        return original(path, value)

    monkeypatch.setattr(module, "_write", interrupt)
    declaration = phases(phases=[{"id": "a", "budget": 5, "analysis": {"sample_interval": 2}}]) if phased else experiment("sig.demo", budget=5, analysis={"sample_interval": 2})
    with pytest.raises(KeyboardInterrupt):
        run(declaration, directory=tmp_path)
    study = Run(next(tmp_path.iterdir()))
    assert study.verify()["valid"]
    monkeypatch.setattr(module, "_write", original)
    resume(study.path)
    assert study.status()["status"] == "complete"
    if phased:
        assert [s["phase_position"] for s in phase_result(study)["samples"]] == [0, 2, 4, 5]
    else:
        trial = study.status()["trials"][0]
        assert [s["position"] for s in read(study.path / trial["path"] / "trajectory.json")["samples"]] == [0, 2, 4]


def test_worker_count_preserves_phase_seeds_and_states(tmp_path):
    serial = run(phases(seeds=[1, 2]), directory=tmp_path / "serial")
    parallel = run(phases(seeds=[1, 2], workers=2), directory=tmp_path / "parallel")
    for trial in range(2):
        for phase in range(2):
            assert checkpoint(serial, phase, trial)["state"] == checkpoint(parallel, phase, trial)["state"]
    assert parallel.verify()["valid"]


def test_phase_checkpoint_tampering_is_rejected(tmp_path):
    study = run(phases(phases=[{"id": "a", "budget": 1}]), directory=tmp_path)
    Path(phase_result(study)["checkpoint"]).write_text("{}")
    assert not study.verify()["valid"]
    with pytest.raises(ValueError, match="verification"):
        resume(study.path)


def rewrite_receipts(directory, transform):
    """Rehash valid documents so tests exercise declaration binding itself."""
    previous = None
    for path in sorted((directory / "commits").glob("*/receipt.json")):
        receipt = read(path)
        checkpoint_path = path.parent / "checkpoint.json"
        state = read(checkpoint_path)
        transform(receipt, state)
        state.pop("checkpoint_hash")
        state["checkpoint_hash"] = digest(state)
        checkpoint_path.write_text(json.dumps(state))
        receipt["checkpoint_file_hash"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        receipt["previous_receipt_hash"] = previous
        receipt.pop("receipt_hash")
        receipt["receipt_hash"] = digest(receipt)
        path.write_text(json.dumps(receipt))
        previous = receipt["receipt_hash"]


@pytest.mark.parametrize("retag_trial", [False, True])
def test_valid_phase_journal_cannot_be_transplanted_between_seeds(tmp_path, retag_trial):
    study = run(phases(seeds=[1, 2], phases=[{"id": "a", "budget": 2}]), directory=tmp_path)
    source = Path(phase_result(study, trial=0)["checkpoint"]).parents[2]
    destination = Path(phase_result(study, trial=1)["checkpoint"]).parents[2]
    shutil.rmtree(destination)
    shutil.copytree(source, destination)
    if retag_trial:
        rewrite_receipts(destination, lambda receipt, state: receipt.update(trial_seed=2))
    assert not study.verify()["valid"]
    with pytest.raises(ValueError, match="verification"):
        resume(study.path)


@pytest.mark.parametrize("change", ["drop", "duplicate", "seed", "path"])
def test_manifest_inventory_must_match_all_declared_trials(tmp_path, change):
    study = run(phases(seeds=[1, 2], phases=[{"id": "a", "budget": 1}]), directory=tmp_path)
    manifest = read(study.path / "manifest.json")
    if change == "drop":
        manifest["trials"].pop()
    elif change == "duplicate":
        manifest["trials"][1] = manifest["trials"][0]
    elif change == "seed":
        manifest["trials"][1]["seed"] = 9
    else:
        manifest["trials"][1]["path"] = manifest["trials"][0]["path"]
    (study.path / "manifest.json").write_text(json.dumps(manifest))
    report = study.verify()
    assert not report["valid"] and "inventory" in report["errors"][0]
    with pytest.raises(ValueError, match="inventory"):
        study.status()
    with pytest.raises(ValueError, match="inventory"):
        study.evaluate()


@pytest.mark.parametrize("change", ["phase", "protocol", "interventions", "start-kind", "transition", "start-position", "seed"])
def test_carry_receipts_are_bound_to_the_declared_operation(tmp_path, change):
    study = run(phases(), directory=tmp_path)
    directory = Path(phase_result(study, 1)["checkpoint"]).parents[2]

    def alter(receipt, state):
        if change == "phase":
            receipt["phase"] = "another-phase"
        elif change == "protocol":
            state["protocol"] = {"window": 7}
        elif change == "interventions":
            receipt["interventions"] = [{"kind": "trainability", "modules": ["sender"], "enabled": False}]
        elif change == "start-kind":
            receipt["inputs"]["kind"] = "fresh"
        elif change == "transition":
            receipt["inputs"]["transition"] = {"kind": "release_fixed", "fixed_weight": 9, "other_weight": 1}
        elif change == "start-position":
            receipt["start_position"] = 0
            receipt["phase_position"] = receipt["position"]
        else:
            receipt["phase_seed"] = state["seed"] = 999

    rewrite_receipts(directory, alter)
    assert not study.verify()["valid"]


def test_trainability_intervention_preserves_frozen_policy(tmp_path):
    study = run(phases(phases=[
        {"id": "a", "budget": 3},
        {"id": "b", "budget": 4, "start": {"kind": "carry", "phase": "a"}, "interventions": [
            {"kind": "trainability", "modules": ["sender"], "enabled": False},
        ]},
    ]), directory=tmp_path)
    source, child = checkpoint(study), checkpoint(study, 1)
    for key, table in source["state"]["runtime"]["tables"].items():
        if table["module"] == "sender":
            after = child["state"]["runtime"]["tables"][key]
            assert after["weights"] == table["weights"] and after["frozen"]
    assert study.verify()["valid"]


@pytest.mark.parametrize("target,bindings", [
    ("tr-log.demo", {"template": {"phase": "nand"}}),
    ("comp-nand.2.demo", {"first": {"phase": "nand"}, "second": {"phase": "nand"}}),
])
def test_compose_archives_components_and_preserves_source_and_frozen_tables(tmp_path, target, bindings):
    study = run(phases(phases=[
        {"id": "nand", "preset": "nand.demo", "budget": 3},
        {"id": "composed", "preset": target, "budget": 2, "start": {"kind": "compose", "bindings": bindings}},
    ]), directory=tmp_path)
    assert study.status()["status"] == "complete"
    source = checkpoint(study)
    result = phase_result(study, 1)
    composed = checkpoint(study, 1)
    assert source["state"]["runtime"]["clocks"]["play"] == 3
    assert composed["state"]["runtime"]["clocks"]["play"] == 2
    assert composed["seed"] == phase_seed(17, "composed")
    assert result["effective_model"]["interaction"]["parameters"]["template_mode"] == "artifact"
    for slot, artifact in result["inputs"]["artifacts"].items():
        path = Path(result["checkpoint"]).parents[2] / artifact["path"]
        exported = read(path)
        assert exported["source"]["checkpoint_hash"] == source["checkpoint_hash"]
        assert exported["source"]["position"] == 3
        assert artifact["parent_receipt_hash"] == phase_result(study)["receipt_hash"]
        for row in composed["state"]["runtime"]["tables"].values():
            if row["module"].startswith(slot + ":"):
                assert row["frozen"]
    assert study.verify()["valid"]
    before = Path(result["checkpoint"]).read_bytes()
    resume(study.path)
    assert Path(result["checkpoint"]).read_bytes() == before


def test_legacy_measurement_retry_keeps_boundary_and_endpoint_crossing(tmp_path, monkeypatch):
    original = System.evaluate
    failed = False

    def evaluate(self, **kwargs):
        nonlocal failed
        if self.position == 2 and not failed:
            failed = True
            raise ValueError("temporary measurement failure")
        value = original(self, **kwargs)
        value["success_probability"] = float(self.position == 5)
        return value

    monkeypatch.setattr(System, "evaluate", evaluate)
    study = run(experiment("sig.demo", budget=5, analysis={"sample_interval": 2, "threshold": 0.9}), directory=tmp_path)
    assert study.status()["status"] == "complete_with_failures"
    resume(study.path)
    trial = study.status()["trials"][0]
    trajectory = study.path / trial["path"] / "trajectory.json"
    assert [s["position"] for s in read(trajectory)["samples"]] == [0, 2, 4]
    assert read(study.path / "results.json")["trials"][0]["threshold"]["first_observed_position"] == 5
    resume(study.path, additional_budget=2)
    assert read(study.path / "results.json")["trials"][0]["threshold"]["first_observed_position"] == 5
    assert study.verify()["valid"]


def test_legacy_stale_manifest_recovers_from_committed_checkpoint(tmp_path):
    study = run(experiment("sig.demo", budget=3, analysis={"sample_interval": 1}), directory=tmp_path)
    manifest = study.status()
    trial = manifest["trials"][0]
    trial.update(status="pending", position=0)
    manifest["status"] = "running"
    # Simulate a manager that retained only initial hashes before termination.
    manifest["hashes"] = {k: v for k, v in manifest["hashes"].items() if k in {"config.toml", "resolved.toml", "resolved.json"}}
    (study.path / "manifest.json").write_text(json.dumps(manifest))
    assert study.verify()["valid"]
    before = read(study.path / trial["path"] / "checkpoint.json")
    resume(study.path)
    assert read(study.path / trial["path"] / "checkpoint.json") == before
    assert study.status()["trials"][0]["position"] == 3
