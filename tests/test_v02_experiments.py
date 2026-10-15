"""Scientific identity and replay through the TOML/workspace entry points."""

import copy
import json
import tomllib

import pytest

from self_assembling_games.config import dumps, from_dict, load, save
from self_assembling_games.domain import Model, experiment, model
from self_assembling_games.experiments import Run, Workspace, resume, run
from self_assembling_games.system import System
from self_assembling_games.cli import main


def checkpoint(study, index=0):
    trial = study.status()["trials"][index]
    return json.loads((study.path / trial["path"] / "checkpoint.json").read_text())


def test_toml_precedence_and_deep_roundtrip(tmp_path):
    source = """schema_version = 2
preset = "sig.demo"
seeds = [2, 4]
budget = 3
[overrides.execution.parameters]
window = 7
[[conditions]]
name = "control"
[[conditions]]
name = "short-window"
budget = 2
[conditions.overrides.execution.parameters]
window = 2
"""
    path = tmp_path / "study.toml"
    path.write_text(source)
    resolved = load(path)
    data = resolved.to_dict()
    assert [c["budget"] for c in data["conditions"]] == [3, 2]
    assert [
        c["model"]["execution"]["parameters"]["window"] for c in data["conditions"]
    ] == [7, 2]
    assert from_dict(tomllib.loads(dumps(resolved))).to_dict() == data
    assert path.read_text() == source
    save(resolved, tmp_path / "resolved.toml")
    assert load(tmp_path / "resolved.toml") == resolved


def test_null_values_survive_resolved_model_toml(tmp_path):
    value = model("ecg.demo").to_dict()
    value["resources"]["extensions"] = {
        "null": None,
        "array": [None, "", None],
        "a/b~c": None,
    }
    value = Model.from_dict(value)
    path = save(value, tmp_path / "model.toml")
    assert load(path).to_dict() == value.to_dict()
    assert "toml_nulls" in path.read_text()


@pytest.mark.parametrize(
    "declaration",
    [
        {"preset": "sig.demo", "bugdet": 2},
        {"preset": "sig.demo", "seeds": [0, 0]},
        {"preset": "sig.demo", "budget": 0.5},
        {"preset": "sig.demo", "workers": 0},
        {"preset": "sig.demo", "conditions": []},
        {"preset": "sig.demo", "conditions": [{"name": "x"}, {"name": "x"}]},
        {"preset": "sig.demo", "conditions": [{"budgett": 3}]},
    ],
)
def test_invalid_studies_fail_before_execution(declaration):
    with pytest.raises((ValueError, TypeError)):
        from_dict(declaration)


def test_one_study_archives_all_conditions_seeds_and_exact_input(tmp_path):
    path = tmp_path / "study.toml"
    original = '# Retained original comment\npreset="sig.demo"\nseeds=[3,5]\nbudget=3\n[[conditions]]\nname="a"\n[[conditions]]\nname="b"\n'
    path.write_text(original)
    study = run(path, directory=tmp_path / "runs")
    assert len(list((tmp_path / "runs").iterdir())) == 1
    assert (study.path / "config.toml").read_text() == original
    assert len(study.status()["trials"]) == 4
    assert {t["position"] for t in study.status()["trials"]} == {3}
    assert study.verify()["valid"]
    summary = json.loads((study.path / "results.json").read_text())["summary"]
    assert summary["a"]["replicates"] == 2
    assert summary["a"]["failed"] == 0
    assert summary["a"]["metrics"]["success_probability"]["n"] == 2


def test_archived_resume_and_evaluation_preserve_trajectory(tmp_path, monkeypatch):
    specification = model("sig.demo")
    study = run(experiment(specification, seeds=[19], budget=4), directory=tmp_path)
    before = checkpoint(study)
    study.evaluate()
    assert checkpoint(study) == before
    resumed = resume(study.path, additional_budget=5)
    expected = specification.instantiate(seed=19).advance(9)
    assert checkpoint(resumed)["state"] == expected.snapshot()["state"]
    assert resumed.status()["trials"][0]["position"] == 9
    assert resumed.status()["resumptions"][0]["additional_budget"] == 5
    # A second complete resume is idempotent and uses the archived declaration.
    import self_assembling_games.domain as domain

    monkeypatch.setattr(
        domain,
        "model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live preset lookup")
        ),
    )
    unchanged = copy.deepcopy(checkpoint(resumed))
    resume(resumed.path)
    assert checkpoint(resumed) == unchanged


def test_worker_count_preserves_seed_trajectories(tmp_path):
    sequential = run(
        experiment("sig.demo", seeds=[6, 9], budget=4), directory=tmp_path / "serial"
    )
    parallel = run(
        experiment("sig.demo", seeds=[6, 9], budget=4, workers=2),
        directory=tmp_path / "parallel",
    )
    assert [checkpoint(sequential, i)["state"] for i in range(2)] == [
        checkpoint(parallel, i)["state"] for i in range(2)
    ]
    assert parallel.verify()["valid"]


def test_seed_failure_is_recorded_and_resumes_from_checkpoint(tmp_path, monkeypatch):
    original = System.evaluate

    def fail_one(self, **parameters):
        if self.seed == 7:
            raise ValueError("intentional evaluation interruption")
        return original(self, **parameters)

    monkeypatch.setattr(System, "evaluate", fail_one)
    study = run(experiment("sig.demo", seeds=[7, 8], budget=3), directory=tmp_path)
    assert study.status()["status"] == "complete_with_failures"
    assert [t["status"] for t in study.status()["trials"]] == ["failed", "complete"]
    assert "intentional evaluation" in study.status()["trials"][0]["error"]
    assert study.verify()["valid"]
    retained = checkpoint(study)
    monkeypatch.setattr(System, "evaluate", original)
    resume(study.path)
    assert study.status()["status"] == "complete"
    assert checkpoint(study) == retained


def test_verify_detects_config_and_checkpoint_tampering(tmp_path):
    study = run(experiment("sig.demo", budget=2), directory=tmp_path)
    (study.path / "resolved.toml").write_text("changed=true\n")
    assert not study.verify()["valid"]
    with pytest.raises(ValueError, match="verification failed"):
        resume(study.path)


def test_cli_workspace_lifecycle(tmp_path, capsys):
    assert main(["init", str(tmp_path), "--budget", "3"]) == 0
    capsys.readouterr()
    config = tmp_path / "experiment.toml"
    assert main(["resolve", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "experiment"
    assert main(["run", str(config), "--workspace", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    study = next(
        path.parent for path in Workspace(tmp_path).runs.glob("*/manifest.json")
    )
    assert main(["status", str(study)]) == 0
    capsys.readouterr()
    assert main(["evaluate", str(study)]) == 0
    assert len(json.loads(capsys.readouterr().out)["trials"]) == 1
    assert main(["resume", str(study), "--additional-budget", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["trials"][0]["position"] == 5
    assert main(["verify", str(study)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert Run.load(study).verify()["valid"]


def test_trajectory_bootstrap_and_threshold_counts(tmp_path):
    study = run(
        experiment(
            "sig.demo",
            seeds=[2, 5],
            budget=5,
            analysis={
                "sample_interval": 2,
                "bootstrap_samples": 40,
                "bootstrap_seed": 11,
                "threshold_metric": "success_probability",
                "threshold": 2.0,
            },
        ),
        directory=tmp_path,
    )
    result = json.loads((study.path / "results.json").read_text())
    for trial in study.status()["trials"]:
        samples = json.loads(
            (study.path / trial["path"] / "trajectory.json").read_text()
        )["samples"]
        assert [sample["position"] for sample in samples] == [0, 2, 4]
    summary = result["summary"]["default"]
    assert len(summary["metrics"]["success_probability"]["bootstrap_interval"]) == 2
    assert summary["threshold"]["reached"] == 0
    assert summary["threshold"]["budget_censored"] == 2
    assert summary["threshold"]["wilson_interval"][1] > 0
    assert study.verify()["valid"]
    resume(study.path, additional_budget=2)
    samples = json.loads(
        (
            study.path / study.status()["trials"][0]["path"] / "trajectory.json"
        ).read_text()
    )["samples"]
    assert [sample["position"] for sample in samples] == [0, 2, 4, 6]


@pytest.mark.parametrize(
    "analysis",
    [
        {"sample_interval": 0},
        {"sample_interval": 0.5},
        {"bootstrap_samples": 0},
        {"confidence": 1},
        {"threshold": float("inf")},
    ],
)
def test_invalid_analysis_has_no_run_side_effects(tmp_path, analysis):
    with pytest.raises(ValueError):
        run(experiment("sig.demo", budget=1, analysis=analysis), directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_default_sampled_evaluation_for_dynamic_repertoire(tmp_path):
    study = run(
        experiment("inv.demo", budget=0, analysis={"evaluation": {"samples": 3}}),
        directory=tmp_path,
    )
    assert study.status()["status"] == "complete"
    result = json.loads((study.path / "results.json").read_text())
    assert result["trials"][0]["evaluation"]["protocol"] == "sampled"


def test_run_evaluate_inherits_archived_protocol_and_accepts_overrides(tmp_path):
    study = run(
        experiment(
            "sig.demo",
            budget=2,
            analysis={"evaluation": {"protocol": "sampled", "samples": 3, "seed": 18}},
        ),
        directory=tmp_path,
    )
    inherited = study.evaluate()["trials"][0]["evaluation"]
    assert inherited["protocol"] == "sampled"
    assert inherited["attempts"] == 3
    overridden = study.evaluate(samples=7)["trials"][0]["evaluation"]
    assert overridden["attempts"] == 7
    assert study.verify()["valid"]


def test_failed_construction_retains_seed_and_can_retry(tmp_path, monkeypatch):
    original = Model.instantiate

    def fail(*args, **kwargs):
        raise ValueError("temporary construction failure")

    monkeypatch.setattr(Model, "instantiate", fail)
    study = run(experiment("sig.demo", seeds=[17], budget=2), directory=tmp_path)
    assert study.status()["trials"][0]["seed"] == 17
    assert study.verify()["valid"]
    monkeypatch.setattr(Model, "instantiate", original)
    resume(study.path)
    assert study.status()["status"] == "complete"
    assert checkpoint(study)["seed"] == 17
