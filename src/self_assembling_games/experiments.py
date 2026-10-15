"""Archived studies: one directory, condition/seed trials, and resumable state."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import uuid

from .config import dumps


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "condition"


def _recovery_receipts(directory):
    """Legacy trial journal; the final receipt commits its immutable checkpoint."""
    from .simulation import digest

    receipts, previous = [], None
    for path in sorted((Path(directory) / "recovery").glob("*/receipt.json")):
        receipt = _read(path)
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_hash"}
        if digest(unsigned) != receipt.get("receipt_hash") or receipt.get("previous_receipt_hash") != previous:
            raise ValueError("Trial recovery receipt integrity mismatch")
        checkpoint = path.parent / "checkpoint.json"
        if not checkpoint.is_file() or _hash(checkpoint) != receipt["checkpoint_file_hash"]:
            raise ValueError("Trial recovery checkpoint changed or missing")
        receipts.append((path, receipt))
        previous = receipt["receipt_hash"]
    return receipts


def _retain_trial(directory, system, previous, *, pending=None, measurement=None, historical=None):
    from .simulation import digest

    directory = Path(directory)
    recovery = directory / "recovery"
    recovery.mkdir(exist_ok=True)
    index = previous.get("sequence", -1) + 1 if previous else 0
    while (recovery / f"{index:08d}").exists():
        index += 1
    generation = recovery / f"{index:08d}"
    generation.mkdir()
    snapshot = system.snapshot()
    _write(generation / "checkpoint.json", snapshot)
    receipt = {
        "schema_version": 1, "sequence": index, "position": system.position,
        "previous_receipt_hash": previous["receipt_hash"] if previous else None,
        "checkpoint_file_hash": _hash(generation / "checkpoint.json"),
        "pending_measurement": pending, "measurement": measurement,
        "historical_samples": historical or [],
    }
    receipt["receipt_hash"] = digest(receipt)
    _write(generation / "receipt.json", receipt)
    # Compatibility file is derived; a hard stop before its replacement leaves
    # the immutable generation available for recovery.
    _write(directory / "checkpoint.json", snapshot)
    return receipt


def _recovery_observations(receipts):
    return [sample for _, receipt in receipts for sample in (
        receipt.get("historical_samples", [])
        + ([receipt["measurement"]] if receipt.get("measurement") is not None else [])
    )]


def _evaluation_parameters(model, parameters):
    parameters = dict(parameters)
    capabilities = model.get("capabilities", [])
    if (
        "protocol" not in parameters
        and "evaluate.exact" not in capabilities
        and "evaluate.sampled" in capabilities
    ):
        parameters["protocol"] = "sampled"
    return parameters


def _validate_analysis(config):
    analysis = config["analysis"]
    allowed = {
        "evaluation",
        "sample_interval",
        "bootstrap_samples",
        "bootstrap_seed",
        "confidence",
        "threshold",
        "threshold_metric",
    }
    if not isinstance(analysis, dict) or analysis.keys() - allowed:
        raise ValueError(
            "Unknown analysis fields; use evaluation, sample_interval, bootstrap_samples, bootstrap_seed, confidence, threshold or threshold_metric"
        )
    if not isinstance(analysis.get("evaluation", {}), dict):
        raise ValueError("analysis.evaluation must be a table")
    for key in ("sample_interval", "threshold"):
        if key in analysis and (
            type(analysis[key]) not in (int, float) or not math.isfinite(analysis[key])
        ):
            raise ValueError(f"analysis.{key} must be finite")
    if "sample_interval" in analysis:
        interval = analysis["sample_interval"]
        if interval <= 0 or (
            any(
                c["model"]["execution"]["clock"] != "time" for c in config["conditions"]
            )
            and type(interval) is not int
        ):
            raise ValueError(
                "Sample interval must be positive and integer for discrete clocks"
            )
    for key, minimum in (("bootstrap_samples", 1), ("bootstrap_seed", 0)):
        if key in analysis and (
            type(analysis[key]) is not int or analysis[key] < minimum
        ):
            raise ValueError(f"analysis.{key} must be an integer >= {minimum}")
    confidence = analysis.get("confidence", 0.95)
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("analysis.confidence must lie between zero and one")
    if "threshold_metric" in analysis and not isinstance(
        analysis["threshold_metric"], str
    ):
        raise ValueError("analysis.threshold_metric must be a metric name")


@dataclass(frozen=True)
class Workspace:
    """An explicit destination for study directories and portable model TOMLs."""

    path: Path

    def __init__(self, path):
        object.__setattr__(self, "path", Path(path))

    @property
    def runs(self):
        return self.path

    @property
    def models(self):
        return self.path / "models"

    def init(self):
        self.runs.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        return self

    def run(self, experiment):
        return run(experiment, workspace=self)


def workspace(path="runs/default") -> Workspace:
    return Workspace(path).init()


@dataclass(frozen=True)
class Run:
    path: Path

    def __init__(self, path):
        object.__setattr__(self, "path", Path(path))

    @classmethod
    def load(cls, path):
        result = cls(path)
        result._manifest()
        return result

    def _manifest(self):
        data = _read(self.path / "manifest.json")
        if data.get("schema_version") not in (2, 3) or data.get("kind") != "run":
            raise ValueError("Expected a schema 2 or 3 run directory")
        return data

    def status(self):
        if self._manifest()["schema_version"] == 3:
            from .phase_experiments import status

            return status(self.path)
        manifest = self._manifest()
        for trial in manifest["trials"]:
            journal = _recovery_receipts(self.path / trial["path"])
            if journal and (journal[-1][1]["position"] != trial.get("position") or (journal[-1][1]["pending_measurement"] is not None and trial["status"] != "failed")):
                trial.update(position=journal[-1][1]["position"], status="interrupted")
                manifest["status"] = "interrupted"
        return manifest

    def evaluate(self, **parameters):
        """Evaluate checkpoint copies; persisted trajectory and RNG state stay intact."""
        if self._manifest()["schema_version"] == 3:
            from .phase_experiments import evaluate

            return evaluate(self.path, **parameters)
        from .system import System

        archived = _read(self.path / "resolved.json")["analysis"].get("evaluation", {})
        parameters = {**archived, **parameters}
        results = []
        for trial in self._manifest()["trials"]:
            checkpoint = self.path / trial["path"] / "checkpoint.json"
            journal = _recovery_receipts(self.path / trial["path"])
            if journal:
                checkpoint = journal[-1][0].parent / "checkpoint.json"
            if checkpoint.exists():
                system = System.restore(_read(checkpoint))
                results.append(
                    {
                        "condition": trial["condition"],
                        "seed": trial["seed"],
                        "evaluation": system.evaluate(
                            **_evaluation_parameters(system.model.to_dict(), parameters)
                        ),
                    }
                )
        return {"path": str(self.path), "trials": results}

    def verify(self):
        """Check archived file identities and the manifest/checkpoint clock agreement."""
        if self._manifest()["schema_version"] == 3:
            from .phase_experiments import verify

            return verify(self.path)
        from .system import System

        manifest = self._manifest()
        errors = []
        recoveries = {}
        for trial in manifest["trials"]:
            try:
                recoveries[trial["path"]] = _recovery_receipts(self.path / trial["path"])
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(f"Invalid recovery journal {trial['path']}: {error}")
        for relative, expected in manifest["hashes"].items():
            path = self.path / relative
            if not path.is_file() or _hash(path) != expected:
                trial_path, _, name = relative.rpartition("/")
                journal = recoveries.get(trial_path, [])
                if manifest["status"] in {"running", "interrupted"} and name == "results.json" and journal:
                    continue  # Derived trial report can lag its committed state.
                if manifest["status"] in {"running", "interrupted"} and relative in {"results.json", "results/summary.json"}:
                    continue
                if journal and name == "checkpoint.json" and path.is_file() and _hash(path) in {r["checkpoint_file_hash"] for _, r in journal}:
                    continue
                if journal and name == "trajectory.json" and path.is_file():
                    retained = _recovery_observations(journal)
                    if all(sample in retained for sample in _read(path).get("samples", [])):
                        continue
                errors.append(f"File changed or missing: {relative}")
        for trial in manifest["trials"]:
            checkpoint = self.path / trial["path"] / "checkpoint.json"
            if not checkpoint.exists():
                if not (
                    trial["status"] in {"pending", "failed"}
                    and trial.get("position", 0) == 0
                ):
                    errors.append(f"Missing checkpoint: {trial['path']}")
                continue
            try:
                journal = recoveries.get(trial["path"], [])
                system = System.restore(_read(journal[-1][0].parent / "checkpoint.json" if journal else checkpoint))
                if journal:
                    if system.position != journal[-1][1]["position"]:
                        errors.append(f"Recovery checkpoint position differs: {trial['path']}")
                    if system.position > trial["budget"] + 1e-10:
                        errors.append(f"Recovery checkpoint exceeds budget: {trial['path']}")
                elif system.position != trial.get("position"):
                    errors.append(f"Checkpoint position differs: {trial['path']}")
            except (ValueError, KeyError, TypeError) as error:
                errors.append(f"Invalid checkpoint {trial['path']}: {error}")
        return {"path": str(self.path), "valid": not errors, "errors": errors}

    def resume(self, additional_budget=None):
        return resume(self.path, additional_budget=additional_budget)


def _execute_trial(root, trial, condition, analysis):
    try:
        return _execute_trial_inner(root, trial, condition, analysis)
    except Exception as error:
        from .system import System

        directory = Path(root) / trial["path"]
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "checkpoint.json"
        position = trial.get("position", 0)
        if checkpoint.exists():
            try:
                position = System.restore(_read(checkpoint)).position
            except (ValueError, KeyError, TypeError):
                pass
        result = {
            "condition": trial["condition"],
            "seed": trial["seed"],
            "status": "failed",
            "position": position,
            "budget": trial["budget"],
            "error": f"{type(error).__name__}: {error}",
        }
        _write(directory / "results.json", result)
        return result


def _execute_trial_inner(root, trial, condition, analysis):
    from .domain import Model
    from .system import System

    root = Path(root)
    directory = root / trial["path"]
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"
    journal = _recovery_receipts(directory)
    if journal:
        system = System.restore(_read(journal[-1][0].parent / "checkpoint.json"))
        previous = journal[-1][1]
    elif checkpoint.exists():
        system = System.restore(_read(checkpoint))
        previous = None
    else:
        system = Model.from_dict(condition["model"]).instantiate(
            seed=trial["seed"], protocol=condition["protocol"]
        )
        previous = None
    if system.model.to_dict() != condition["model"] or system.seed != trial["seed"] or system.protocol != condition["protocol"]:
        raise ValueError("Trial checkpoint differs from its archived declaration")
    trajectory_path = directory / "trajectory.json"
    samples = _read(trajectory_path)["samples"] if trajectory_path.exists() else []
    interval = analysis.get("sample_interval")
    evaluation_parameters = _evaluation_parameters(
        condition["model"], analysis.get("evaluation", {})
    )
    if previous is None:
        historical = list(samples)
        old_results = directory / "results.json"
        if old_results.exists():
            value = _read(old_results)
            if "evaluation" in value:
                historical.append({"position": value["position"], "evaluation": value["evaluation"]})
        pending = "interval" if interval is not None and system.position == 0 and not samples else None
        previous = _retain_trial(directory, system, None, pending=pending, historical=historical)

    def measure(kind):
        nonlocal previous
        sample = {"position": system.position, "evaluation": system.evaluate(**evaluation_parameters)}
        if kind == "interval":
            # Replaying a measurement after a hard stop is idempotent.
            samples[:] = [s for s in samples if s["position"] != system.position]
            samples.append(sample)
            _write(trajectory_path, {"samples": samples})
        previous = _retain_trial(directory, system, previous, measurement=sample)
        return sample["evaluation"]

    # Complete an interrupted boundary measurement before the next advance.
    if previous["pending_measurement"] is not None:
        measure(previous["pending_measurement"])
    remaining = trial["budget"] - system.position
    if remaining < -1e-10:
        raise ValueError("Checkpoint is beyond the archived trial budget")
    while remaining > 1e-10:
        amount = remaining if interval is None else min(
            remaining,
            (math.floor(system.position / interval + 1e-10) + 1) * interval - system.position,
        )
        if system.clock != "time":
            amount = int(amount)
        before = system.position
        system.advance(amount)
        on_interval = interval is not None and math.isclose(
            system.position / interval, round(system.position / interval), abs_tol=1e-8
        )
        kind = "interval" if on_interval else "endpoint"
        previous = _retain_trial(directory, system, previous, pending=kind)
        measure(kind)
        remaining = trial["budget"] - system.position
        if system.position <= before:
            break
    observations = _recovery_observations(_recovery_receipts(directory))
    if observations and observations[-1]["position"] == system.position:
        evaluation = observations[-1]["evaluation"]
    else:
        evaluation = measure("endpoint")
        observations = _recovery_observations(_recovery_receipts(directory))
    results = {
        "condition": trial["condition"], "seed": trial["seed"],
        "clock": system.clock, "position": system.position,
        "budget": trial["budget"], "status": system.status,
        "evaluation": evaluation,
    }
    if "threshold" in analysis:
        metric, target = analysis.get("threshold_metric", "success_probability"), analysis["threshold"]
        if metric not in evaluation:
            raise ValueError(f"Threshold metric unavailable: {metric}")
        reached = [sample["position"] for sample in observations
                   if type(sample["evaluation"].get(metric)) in (int, float)
                   and sample["evaluation"][metric] >= target]
        results["threshold"] = {
            "metric": metric, "target": target, "reached": bool(reached),
            "first_observed_position": min(reached) if reached else None,
            "budget_censored": not reached and system.position >= trial["budget"],
        }
    _write(directory / "results.json", results)
    return results


def _summary(results, analysis):
    """Descriptive replicate summaries keep errors and censoring visible."""
    import numpy as np

    groups = {}
    rng = np.random.default_rng(analysis.get("bootstrap_seed", 0))
    confidence = analysis.get("confidence", 0.95)
    for result in results:
        groups.setdefault(result["condition"], []).append(result)
    summaries = {}
    for name, trials in groups.items():
        measured = [trial for trial in trials if "evaluation" in trial]
        keys = set().union(*(trial["evaluation"].keys() for trial in measured))
        metrics = {}
        for key in sorted(keys):
            values = [
                trial["evaluation"][key]
                for trial in measured
                if type(trial["evaluation"].get(key)) in (int, float)
                and math.isfinite(trial["evaluation"][key])
            ]
            if values:
                metrics[key] = {
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "standard_deviation": statistics.stdev(values)
                    if len(values) > 1
                    else None,
                    "minimum": min(values),
                    "maximum": max(values),
                }
                if len(values) > 1:
                    means = [
                        float(
                            np.mean(rng.choice(values, size=len(values), replace=True))
                        )
                        for _ in range(analysis.get("bootstrap_samples", 1000))
                    ]
                    metrics[key]["bootstrap_interval"] = np.quantile(
                        means, [(1 - confidence) / 2, (1 + confidence) / 2]
                    ).tolist()
                else:
                    metrics[key]["bootstrap_interval"] = None
        statuses = {}
        for trial in trials:
            statuses[trial["status"]] = statuses.get(trial["status"], 0) + 1
        summaries[name] = {
            "replicates": len(trials),
            "evaluated": len(measured),
            "failed": sum(t["status"] == "failed" for t in trials),
            "statuses": statuses,
            "metrics": metrics,
        }
        thresholds = [trial["threshold"] for trial in measured if "threshold" in trial]
        if thresholds:
            from statistics import NormalDist

            n, successes = len(thresholds), sum(t["reached"] for t in thresholds)
            z = NormalDist().inv_cdf((1 + confidence) / 2)
            p, denominator = successes / n, 1 + z * z / n
            centre = (p + z * z / (2 * n)) / denominator
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
            summaries[name]["threshold"] = {
                "evaluated": n,
                "reached": successes,
                "budget_censored": sum(t["budget_censored"] for t in thresholds),
                "wilson_interval": [centre - half, centre + half],
            }
    return summaries


def _archive_hashes(root, manifest):
    for trial in manifest["trials"]:
        for name in ("checkpoint.json", "results.json", "trajectory.json"):
            relative = f"{trial['path']}/{name}"
            if (root / relative).is_file():
                manifest["hashes"][relative] = _hash(root / relative)


def _complete(study):
    root = study.path
    manifest = study._manifest()
    config = _read(root / "resolved.json")
    _validate_analysis(config)
    conditions = {condition["name"]: condition for condition in config["conditions"]}
    todo = [trial for trial in manifest["trials"] if trial["status"] != "complete"]
    manifest["status"] = "running"
    manifest["updated_at"] = _stamp()
    _write(root / "manifest.json", manifest)

    def record(trial, result):
        trial.update(
            status="failed" if result["status"] == "failed" else "complete",
            position=result["position"],
            system_status=result["status"],
        )
        if "error" in result:
            trial["error"] = result["error"]
        else:
            trial.pop("error", None)
        _archive_hashes(root, manifest)
        manifest["updated_at"] = _stamp()
        _write(root / "manifest.json", manifest)

    try:
        workers = config["workers"]
        if workers == 1:
            for trial in todo:
                record(
                    trial,
                    _execute_trial(
                        str(root),
                        trial,
                        conditions[trial["condition"]],
                        config["analysis"],
                    ),
                )
        elif todo:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _execute_trial,
                        str(root),
                        trial,
                        conditions[trial["condition"]],
                        config["analysis"],
                    ): trial
                    for trial in todo
                }
                for future in as_completed(futures):
                    record(futures[future], future.result())
    except BaseException as error:
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(error).__name__}: {error}"
        # A trial can have checkpointed before evaluation or process interruption.
        from .system import System

        for trial in todo:
            checkpoint = root / trial["path"] / "checkpoint.json"
            if checkpoint.exists() and trial["status"] != "complete":
                trial["position"] = System.restore(_read(checkpoint)).position
                trial["status"] = "interrupted"
        _archive_hashes(root, manifest)
        _write(root / "manifest.json", manifest)
        raise
    manifest["status"] = (
        "complete_with_failures"
        if any(t["status"] == "failed" for t in manifest["trials"])
        else "complete"
    )
    manifest.pop("error", None)
    manifest["updated_at"] = _stamp()
    results = [_read(root / t["path"] / "results.json") for t in manifest["trials"]]
    _write(
        root / "results.json",
        {
            "trials": results,
            "summary": _summary(results, config["analysis"]),
            "analysis": config["analysis"],
        },
    )
    (root / "results").mkdir(exist_ok=True)
    _write(root / "results" / "summary.json", _read(root / "results.json"))
    manifest["hashes"]["results.json"] = _hash(root / "results.json")
    manifest["hashes"]["results/summary.json"] = _hash(
        root / "results" / "summary.json"
    )
    _write(root / "manifest.json", manifest)
    return study


def run(experiment, directory="runs", *, workspace=None, original=None) -> Run:
    """Execute all conditions/seeds under one new timestamped study directory."""
    from .config import load
    from .domain import Experiment, experiment as make_experiment

    source = original
    if isinstance(experiment, (str, Path)) and Path(experiment).is_file():
        source = Path(experiment).read_text(encoding="utf-8")
        experiment = load(experiment)
    if not isinstance(experiment, Experiment):
        experiment = make_experiment(experiment)
    config = experiment.to_dict()
    if config["schema_version"] == 3:
        from .phase_experiments import run as run_phases

        return run_phases(experiment, directory=directory, workspace=workspace, original=source)
    _validate_analysis(config)
    destination = (
        workspace.runs
        if isinstance(workspace, Workspace)
        else Path(directory) / workspace
        if workspace is not None
        else Path(directory)
    )
    name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = destination / name
    root.mkdir(parents=True, exist_ok=False)
    resolved = dumps(config)
    (root / "config.toml").write_text(source or resolved, encoding="utf-8")
    (root / "resolved.toml").write_text(resolved, encoding="utf-8")
    _write(root / "resolved.json", config)
    trials = []
    for index, condition in enumerate(config["conditions"]):
        for seed in config["seeds"]:
            trials.append(
                {
                    "condition": condition["name"],
                    "seed": seed,
                    "budget": condition["budget"],
                    "path": f"trials/{index:03d}-{_slug(condition['name'])}/seed-{seed}",
                    "status": "pending",
                    "position": 0,
                }
            )
    from . import __version__

    manifest = {
        "schema_version": 2,
        "kind": "run",
        "status": "pending",
        "created_at": _stamp(),
        "software": {
            "self_assembling_games": __version__,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "trials": trials,
        "hashes": {
            name: _hash(root / name)
            for name in ("config.toml", "resolved.toml", "resolved.json")
        },
        "resumptions": [],
    }
    _write(root / "manifest.json", manifest)
    return _complete(Run(root))


def resume(path, additional_budget=None) -> Run:
    """Continue archived trials, optionally adding an amount to every trial budget."""
    study = Run.load(path)
    if study._manifest()["schema_version"] == 3:
        from .phase_experiments import resume as resume_phases

        return resume_phases(path, additional_budget=additional_budget)
    report = study.verify()
    if not report["valid"]:
        raise ValueError("Run verification failed: " + "; ".join(report["errors"]))
    manifest = study._manifest()
    if additional_budget is not None:
        if (
            isinstance(additional_budget, bool)
            or not isinstance(additional_budget, (int, float))
            or not math.isfinite(additional_budget)
            or additional_budget < 0
        ):
            raise ValueError("Additional budget must be finite and nonnegative")
        config = _read(study.path / "resolved.json")
        if any(
            c["model"]["execution"]["clock"] != "time" for c in config["conditions"]
        ) and not isinstance(additional_budget, int):
            raise ValueError("Discrete clocks require an integer additional budget")
        for trial in manifest["trials"]:
            trial["budget"] += additional_budget
            trial["status"] = "pending"
        manifest["resumptions"].append(
            {"at": _stamp(), "additional_budget": additional_budget}
        )
        _write(study.path / "manifest.json", manifest)
    return _complete(study)
