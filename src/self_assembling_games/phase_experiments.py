"""Ordered phase trials with immutable checkpoint and measurement receipts.

Each receipt is committed last in its generation. Cursors and study manifests
are derived views: a hard stop can leave either behind without losing a complete
checkpoint. Unfinished generation directories are ignored during recovery.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import uuid

from .config import dumps
from .experiments import (
    _evaluation_parameters, _hash, _read, _slug, _stamp, _summary,
    _validate_analysis, _write,
)
from .simulation import digest


def phase_seed(seed, identifier):
    """A stable fresh-system seed, independent of condition order and workers."""
    encoded = json.dumps([seed, identifier], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _validated_analysis(config):
    for condition in config["conditions"]:
        for phase in condition["phases"]:
            _validate_analysis({"analysis": phase["analysis"], "conditions": [{"model": phase["model"]}]})


def _phase_path(root, trial, index, phase):
    return Path(root) / trial["path"] / "phases" / f"{index:03d}-{_slug(phase['id'])}"


def _receipts(directory):
    """Read the complete hash-linked journal; partial generations carry no state."""
    result = []
    previous = None
    for path in sorted((directory / "commits").glob("*/receipt.json")):
        value = _read(path)
        unsigned = {k: v for k, v in value.items() if k != "receipt_hash"}
        if digest(unsigned) != value.get("receipt_hash"):
            raise ValueError(f"Phase receipt integrity mismatch: {path}")
        if value.get("previous_receipt_hash") != previous:
            raise ValueError(f"Phase receipt chain differs: {path}")
        checkpoint = path.parent / "checkpoint.json"
        if not checkpoint.is_file() or _hash(checkpoint) != value["checkpoint_file_hash"]:
            raise ValueError(f"Phase checkpoint changed or missing: {checkpoint}")
        if value.get("sequence") != int(path.parent.name):
            raise ValueError(f"Phase receipt sequence differs: {path}")
        for artifact in value.get("inputs", {}).get("artifacts", {}).values():
            # Relative to the phase directory; filenames are generated internally.
            artifact_path = directory / artifact["path"]
            if not artifact_path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Artifact path escapes its phase directory")
            if not artifact_path.is_file() or _hash(artifact_path) != artifact["sha256"]:
                raise ValueError(f"Phase component changed or missing: {artifact_path}")
        result.append((path, value))
        previous = value["receipt_hash"]
    return result


def _commit(directory, system, previous, context, *, pending=False, measurement=None, complete=False):
    commits = directory / "commits"
    commits.mkdir(parents=True, exist_ok=True)
    sequence = previous["sequence"] + 1 if previous else 0
    while (commits / f"{sequence:08d}").exists():
        sequence += 1
    generation = commits / f"{sequence:08d}"
    generation.mkdir()
    checkpoint = generation / "checkpoint.json"
    _write(checkpoint, system.snapshot())
    value = {
        **copy.deepcopy(context), "schema_version": 1, "sequence": sequence,
        "previous_receipt_hash": previous["receipt_hash"] if previous else None,
        "checkpoint_file_hash": _hash(checkpoint),
        "position": system.position,
        "phase_position": system.position - context["start_position"],
        "system_status": system.status,
        "pending_measurement": pending, "measurement": measurement,
        "complete": complete,
    }
    value["receipt_hash"] = digest(value)
    _write(generation / "receipt.json", value)
    _write(directory / "cursor.json", {"sequence": sequence, "receipt_hash": value["receipt_hash"]})
    return value


def _latest_system(directory, receipts=None):
    from .system import System

    receipts = _receipts(directory) if receipts is None else receipts
    if not receipts:
        return None
    return System.restore(_read(receipts[-1][0].parent / "checkpoint.json"))


def _extensions(root, config):
    previous, total = None, 0
    entries = []
    for path in sorted((Path(root) / "budget_extensions").glob("*.json")):
        entry = _read(path)
        unsigned = {k: v for k, v in entry.items() if k != "extension_hash"}
        if (
            digest(unsigned) != entry.get("extension_hash")
            or entry.get("previous_extension_hash") != previous
        ):
            raise ValueError("Phase budget extension integrity mismatch")
        _validate_extension(entry["additional_budget"], config)
        total += entry["additional_budget"]
        entries.append(entry)
        previous = entry["extension_hash"]
    return entries, total


def _validate_extension(amount, config):
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        raise ValueError("Additional budget must be finite and nonnegative")
    if any(c["phases"][-1]["model"]["execution"]["clock"] != "time" for c in config["conditions"]) and type(amount) is not int:
        raise ValueError("Discrete final phases require an integer additional budget")


def _budget(phase, index, condition, extension):
    return phase["budget"] + (extension if index == len(condition["phases"]) - 1 else 0)


def _trial_inventory(config):
    return [
        {"condition": condition["name"], "seed": seed,
         "path": f"trials/{index:03d}-{_slug(condition['name'])}/seed-{seed}"}
        for index, condition in enumerate(config["conditions"])
        for seed in config["seeds"]
    ]


def _checked_trials(manifest, config):
    expected = _trial_inventory(config)
    if digest(manifest.get("trials")) != digest(expected):
        raise ValueError("Run trial inventory differs from resolved conditions and seeds")
    return expected


def _component_model(declared, templates):
    from .domain import model

    assumptions = [
        assumption for assumption in declared["source"]["assumptions"]
        if not assumption.lower().startswith("constructed perfect")
    ]
    assumptions.append("Frozen components imported from the declared earlier-phase endpoint exports.")
    return model(
        declared,
        interaction={"parameters": {"template_mode": "artifact", "templates": templates}},
        source={"kind": "deliberate_variant", "assumptions": assumptions},
    )


def _initial_system(root, trial, phase, directory, endpoints):
    from .domain import Model
    from .system import System

    start = phase["start"]
    inputs = {"kind": start["kind"], "artifacts": {}}
    seed = phase_seed(trial["seed"], phase["id"])
    target = Model.from_dict(phase["model"])
    if start["kind"] == "carry":
        source_path, source = endpoints[start["phase"]]
        inputs.update(parent_phase=start["phase"], parent_receipt_hash=source["receipt_hash"], transition=start["transition"])
        system = System.restore(_read(source_path)).transition(target, start["transition"])
        if phase["protocol"] and phase["protocol"] != system.protocol:
            raise ValueError("Carry protocol must match the retained System protocol")
        seed = system.seed
    else:
        if start["kind"] == "compose":
            from .component_artifacts import export_component

            templates = {}
            for slot, binding in start["bindings"].items():
                source_path, source = endpoints[binding["phase"]]
                relative = "artifacts/" + hashlib.sha256(slot.encode()).hexdigest() + ".json"
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Export the selected endpoint document verbatim: rebuilding a
                # System snapshot would replace its outer provenance metadata.
                export_component(source_path, destination, specification=binding["export"])
                templates[slot] = str(destination.resolve())
                inputs["artifacts"][slot] = {
                    "path": relative, "sha256": _hash(destination),
                    "parent_phase": binding["phase"],
                    "parent_receipt_hash": source["receipt_hash"],
                    "export": binding["export"],
                }
            target = _component_model(target.to_dict(), templates)
        system = target.instantiate(seed=seed, protocol=phase["protocol"])
    for intervention in phase["interventions"]:
        system.set_trainable(intervention["modules"], intervention["enabled"])
    return system, {
        "condition": trial["condition"], "trial_seed": trial["seed"],
        "phase": phase["id"], "declared_model_hash": digest(phase["model"]),
        "effective_model": system.model.to_dict(), "phase_seed": seed,
        "start_position": system.position, "inputs": inputs,
        "interventions": phase["interventions"],
    }


def _phase_result(directory, phase, budget, receipts=None):
    receipts = _receipts(directory) if receipts is None else receipts
    if not receipts:
        return {"id": phase["id"], "status": "pending", "budget": budget, "position": 0, "phase_position": 0}
    latest = receipts[-1][1]
    samples = [r["measurement"] for _, r in receipts if r["measurement"] is not None]
    complete = latest["complete"] and latest["budget"] == budget
    result = {
        "id": phase["id"], "status": latest["system_status"] if complete else "interrupted",
        "complete": complete, "clock": phase["model"]["execution"]["clock"],
        "position": latest["position"], "phase_position": latest["phase_position"],
        "start_position": latest["start_position"], "budget": budget,
        "budget_reached": latest["phase_position"] >= budget - 1e-10,
        "phase_seed": latest["phase_seed"], "inputs": latest["inputs"],
        "interventions": latest["interventions"],
        "declared_model_hash": latest["declared_model_hash"],
        "effective_model": latest["effective_model"],
        "receipt_hash": latest["receipt_hash"],
        "checkpoint": str(receipts[-1][0].parent / "checkpoint.json"),
        "samples": samples,
    }
    if samples:
        result["evaluation"] = samples[-1]["evaluation"]
    analysis = phase["analysis"]
    if "threshold" in analysis and samples:
        metric, target = analysis.get("threshold_metric", "success_probability"), analysis["threshold"]
        if metric in samples[-1]["evaluation"]:
            reached = [s["phase_position"] for s in samples if type(s["evaluation"].get(metric)) in (int, float) and s["evaluation"][metric] >= target]
            result["threshold"] = {
                "metric": metric, "target": target, "reached": bool(reached),
                "first_observed_position": min(reached) if reached else None,
                "budget_censored": not reached and result["budget_reached"],
            }
    return result


def _execute_phase(root, trial, index, phase, budget, endpoints):
    directory = _phase_path(root, trial, index, phase)
    directory.mkdir(parents=True, exist_ok=True)
    receipts = _receipts(directory)
    interval = phase["analysis"].get("sample_interval")
    if receipts:
        previous = receipts[-1][1]
        if previous["declared_model_hash"] != digest(phase["model"]):
            raise ValueError("Phase checkpoint model differs from its declaration")
        if previous["complete"] and previous["budget"] == budget:
            return _phase_result(directory, phase, budget, receipts)
        system = _latest_system(directory, receipts)
        context = {key: copy.deepcopy(previous[key]) for key in (
            "condition", "trial_seed", "phase", "declared_model_hash", "effective_model", "phase_seed", "start_position", "inputs", "interventions"
        )}
        if system.position != previous["position"]:
            raise ValueError("Phase checkpoint position differs from its receipt")
    else:
        system, context = _initial_system(root, trial, phase, directory, endpoints)
        context["budget"] = budget
        previous = _commit(directory, system, None, context, pending=interval is not None or budget == 0)
    context["budget"] = budget
    evaluation = _evaluation_parameters(system.model.to_dict(), phase["analysis"].get("evaluation", {}))
    while True:
        progress = system.position - context["start_position"]
        if previous["pending_measurement"]:
            # The retained state owns the still-unmeasured boundary. Complete it
            # before moving; failure leaves the same receipt available to retry.
            measured = system.evaluate(**evaluation)
            if "threshold" in phase["analysis"]:
                metric = phase["analysis"].get("threshold_metric", "success_probability")
                if metric not in measured:
                    raise ValueError(f"Threshold metric unavailable: {metric}")
            observation = {
                "position": system.position, "phase_position": progress,
                "evaluation": measured,
            }
            previous = _commit(directory, system, previous, context, measurement=observation)
        if progress >= budget - 1e-10:
            previous = _commit(directory, system, previous, context, complete=True)
            break
        amount = budget - progress
        if interval is not None:
            amount = min(amount, (math.floor(progress / interval + 1e-10) + 1) * interval - progress)
        if system.clock != "time":
            amount = int(amount)
        before = system.position
        system.advance(amount)
        previous = _commit(directory, system, previous, context, pending=True)
        if system.position <= before:
            observation = {"position": system.position, "phase_position": progress, "evaluation": system.evaluate(**evaluation)}
            previous = _commit(directory, system, previous, context, measurement=observation, complete=True)
            break
    result = _phase_result(directory, phase, budget)
    _write(directory / "results.json", result)
    return result


def _execute_trial(task):
    root, trial, condition, extension = task
    endpoints, results = {}, []
    for index, phase in enumerate(condition["phases"]):
        budget = _budget(phase, index, condition, extension)
        directory = _phase_path(root, trial, index, phase)
        try:
            result = _execute_phase(root, trial, index, phase, budget, endpoints)
            receipts = _receipts(directory)
            endpoints[phase["id"]] = (receipts[-1][0].parent / "checkpoint.json", receipts[-1][1])
        except Exception as error:
            # Recoverable training/measurement failures retain the latest complete
            # receipt. Source/configuration tampering is rejected before retry.
            result = _phase_result(directory, phase, budget)
            if "evaluation" in result:
                result["last_observed_evaluation"] = result.pop("evaluation")
            result.pop("threshold", None)
            result.update(status="failed", complete=False, error=f"{type(error).__name__}: {error}")
            results.append(result)
            return {"condition": trial["condition"], "seed": trial["seed"], "status": "failed", "phases": results}
        results.append(result)
    return {"condition": trial["condition"], "seed": trial["seed"], "status": "complete", "phases": results}


def _configuration(root):
    from .domain import Experiment

    config = Experiment.from_dict(_read(Path(root) / "resolved.json")).to_dict()
    _validated_analysis(config)
    return config


def _trial_status(root, trial, condition, extension):
    phases = []
    for index, phase in enumerate(condition["phases"]):
        phases.append(_phase_result(_phase_path(root, trial, index, phase), phase, _budget(phase, index, condition, extension)))
    complete = all(p.get("complete", False) for p in phases)
    return {**trial, "status": "complete" if complete else "interrupted" if any(p["status"] != "pending" for p in phases) else "pending", "phases": phases}


def status(root):
    root = Path(root)
    manifest, config = _read(root / "manifest.json"), _configuration(root)
    entries, extension = _extensions(root, config)
    conditions = {c["name"]: c for c in config["conditions"]}
    trials = [_trial_status(root, t, conditions[t["condition"]], extension) for t in _checked_trials(manifest, config)]
    reports = _read(root / "results.json").get("trials", []) if (root / "results.json").is_file() else []
    failures = {(t["condition"], t["seed"], p["id"]): p for t in reports for p in t["phases"] if p["status"] == "failed"}
    for trial in trials:
        for phase in trial["phases"]:
            failed = failures.get((trial["condition"], trial["seed"], phase["id"]))
            if failed and not phase.get("complete") and failed.get("receipt_hash") == phase.get("receipt_hash"):
                phase.update(status="failed", error=failed["error"])
                trial["status"] = "failed"
    result = {**manifest, "trials": trials, "resumptions": entries}
    if all(t["status"] == "complete" for t in trials):
        result["status"] = "complete"
    elif manifest["status"] == "complete_with_failures":
        result["status"] = "complete_with_failures"
    else:
        result["status"] = "interrupted" if any(t["status"] != "pending" for t in trials) else "pending"
    return result


def _verify_phase_binding(directory, trial, phase, receipt, checkpoint, parents):
    """Bind a valid checkpoint to the trial and operation that produced it."""
    start = phase["start"]
    expected_seed = phase_seed(trial["seed"], phase["id"])
    expected_position = 0
    expected_protocol = phase["protocol"]
    expected_inputs = {"kind": start["kind"], "artifacts": {}}
    expected_model = phase["model"]
    if start["kind"] == "carry":
        parent = parents[start["phase"]]
        if not parent["complete"]:
            raise ValueError("State carry requires a completed parent phase endpoint")
        expected_seed = parent["checkpoint"]["seed"]
        expected_position = parent["position"]
        expected_protocol = parent["checkpoint"]["protocol"]
        if phase["protocol"] and phase["protocol"] != expected_protocol:
            raise ValueError("Carry protocol differs from its declared parent")
        expected_inputs.update(
            parent_phase=start["phase"], parent_receipt_hash=parent["receipt_hash"],
            transition=start["transition"],
        )
    elif start["kind"] == "compose":
        from .component_artifacts import load_component_snapshot

        artifacts = receipt["inputs"].get("artifacts", {})
        if set(artifacts) != set(start["bindings"]):
            raise ValueError("Phase component slots differ from declared bindings")
        templates = receipt["effective_model"]["interaction"]["parameters"].get("templates", {})
        if set(templates) != set(start["bindings"]):
            raise ValueError("Effective model component slots differ from declared bindings")
        for slot, binding in start["bindings"].items():
            parent = parents[binding["phase"]]
            if not parent["complete"]:
                raise ValueError("Composition requires completed parent phase endpoints")
            relative = "artifacts/" + hashlib.sha256(slot.encode()).hexdigest() + ".json"
            expected_inputs["artifacts"][slot] = {
                "path": relative, "sha256": _hash(directory / relative),
                "parent_phase": binding["phase"],
                "parent_receipt_hash": parent["receipt_hash"],
                "export": binding["export"],
            }
            # Absolute template paths describe the original archive location.
            # The receipt's relative, hashed asset remains authoritative after a
            # run directory is moved; retain the original effective model hash.
            origin = templates[slot]
            if not isinstance(origin, str) or Path(origin).name != Path(relative).name or Path(origin).parent.name != "artifacts":
                raise ValueError("Effective model template path differs from its declared slot")
            load_component_snapshot(directory / relative)
            source = _read(directory / relative)["source"]
            selected = parent["checkpoint"]
            if (
                source.get("checkpoint_hash") != selected["checkpoint_hash"]
                or source.get("snapshot_hash") != digest(selected)
                or source.get("model_hash") != selected["model_hash"]
                or source.get("seed") != selected["seed"]
                or source.get("position") != parent["position"]
            ):
                raise ValueError("Component source differs from the selected parent endpoint")
        expected_model = _component_model(phase["model"], templates).to_dict()
    if receipt.get("condition") != trial["condition"] or type(receipt.get("trial_seed")) is not int or receipt["trial_seed"] != trial["seed"]:
        raise ValueError("Phase receipt trial identity differs")
    if receipt["phase"] != phase["id"]:
        raise ValueError("Phase receipt id differs from its declaration")
    if type(receipt["phase_seed"]) is not int or type(checkpoint["seed"]) is not int or receipt["phase_seed"] != expected_seed or checkpoint["seed"] != expected_seed:
        raise ValueError("Phase checkpoint seed differs from its declared trial and parent")
    if digest(checkpoint["protocol"]) != digest(expected_protocol):
        raise ValueError("Phase checkpoint protocol differs from its declaration")
    if digest(receipt["interventions"]) != digest(phase["interventions"]):
        raise ValueError("Phase interventions differ from their declaration")
    if digest(receipt["inputs"]) != digest(expected_inputs):
        raise ValueError("Phase initialization differs from its declaration")
    if receipt["effective_model"] != expected_model or checkpoint["model"] != expected_model:
        raise ValueError("Phase effective model differs from its declaration")
    if receipt["start_position"] != expected_position:
        raise ValueError("Phase starting position differs from its declared parent")
    if receipt["position"] < expected_position or receipt["phase_position"] != receipt["position"] - expected_position:
        raise ValueError("Phase progress differs from its native starting position")
    observation = receipt.get("measurement")
    if observation is not None and (
        observation["position"] != receipt["position"]
        or observation["phase_position"] != receipt["phase_position"]
    ):
        raise ValueError("Phase measurement position differs from its checkpoint")


def verify(root):
    from .system import System

    root = Path(root)
    errors = []
    try:
        manifest, config = _read(root / "manifest.json"), _configuration(root)
        trials = _checked_trials(manifest, config)
        if set(manifest["hashes"]) != {"config.toml", "resolved.toml", "resolved.json"}:
            raise ValueError("Run configuration hash inventory differs")
        for relative, expected in manifest["hashes"].items():
            if not (root / relative).is_file() or _hash(root / relative) != expected:
                errors.append(f"File changed or missing: {relative}")
        extensions, extension = _extensions(root, config)
        conditions = {c["name"]: c for c in config["conditions"]}
        for trial in trials:
            endpoints = {}
            condition = conditions[trial["condition"]]
            for index, phase in enumerate(condition["phases"]):
                directory = _phase_path(root, trial, index, phase)
                receipts = _receipts(directory)
                if not receipts:
                    continue
                budget = _budget(phase, index, condition, extension)
                previous_position = None
                allowed_budgets = {phase["budget"]}
                if index == len(condition["phases"]) - 1:
                    amount = phase["budget"]
                    for entry in extensions:
                        amount += entry["additional_budget"]
                        allowed_budgets.add(amount)
                for path, receipt in receipts:
                    if receipt["declared_model_hash"] != digest(phase["model"]):
                        raise ValueError("Phase receipt declaration differs")
                    if receipt["budget"] not in allowed_budgets or receipt["budget"] > budget:
                        raise ValueError("Phase receipt budget differs from its declared extension history")
                    checkpoint = _read(path.parent / "checkpoint.json")
                    # Restore the latest numerical state; earlier snapshots have
                    # their full System integrity hash checked without rebuilding.
                    unsigned = {k: v for k, v in checkpoint.items() if k != "checkpoint_hash"}
                    if digest(unsigned) != checkpoint.get("checkpoint_hash"):
                        raise ValueError(f"System checkpoint integrity mismatch: {path}")
                    if digest(checkpoint["model"]) != checkpoint["model_hash"]:
                        raise ValueError("Phase checkpoint model integrity mismatch")
                    _verify_phase_binding(directory, trial, phase, receipt, checkpoint, endpoints)
                    if previous_position is not None and receipt["position"] < previous_position:
                        raise ValueError("Phase checkpoint clock moves backwards")
                    previous_position = receipt["position"]
                restored = System.restore(_read(receipts[-1][0].parent / "checkpoint.json"))
                if restored.position != receipts[-1][1]["position"]:
                    raise ValueError("Phase position differs from checkpoint")
                endpoints[phase["id"]] = {
                    "receipt_hash": receipts[-1][1]["receipt_hash"],
                    "position": receipts[-1][1]["position"],
                    "complete": receipts[-1][1]["complete"],
                    "checkpoint": _read(receipts[-1][0].parent / "checkpoint.json"),
                }
    except (ValueError, KeyError, TypeError, OSError) as error:
        errors.append(str(error))
    return {"path": str(root), "valid": not errors, "errors": errors}


def evaluate(root, **parameters):
    root, config = Path(root), _configuration(root)
    manifest = _read(root / "manifest.json")
    conditions = {c["name"]: c for c in config["conditions"]}
    results = []
    for trial in _checked_trials(manifest, config):
        phases = []
        for index, phase in enumerate(conditions[trial["condition"]]["phases"]):
            system = _latest_system(_phase_path(root, trial, index, phase))
            if system is not None:
                options = {**phase["analysis"].get("evaluation", {}), **parameters}
                phases.append({"id": phase["id"], "position": system.position, "evaluation": system.evaluate(**_evaluation_parameters(system.model.to_dict(), options))})
        results.append({"condition": trial["condition"], "seed": trial["seed"], "phases": phases})
    return {"path": str(root), "trials": results}


def _complete(root):
    from .experiments import Run

    root = Path(root)
    config, manifest = _configuration(root), _read(root / "manifest.json")
    _, extension = _extensions(root, config)
    conditions = {c["name"]: c for c in config["conditions"]}
    manifest.update(status="running", updated_at=_stamp())
    _write(root / "manifest.json", manifest)
    tasks = [(str(root), t, conditions[t["condition"]], extension) for t in _checked_trials(manifest, config)]
    try:
        if config["workers"] == 1:
            results = [_execute_trial(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
                results = list(pool.map(_execute_trial, tasks))
    except BaseException:
        manifest.update(status="interrupted", updated_at=_stamp())
        _write(root / "manifest.json", manifest)
        raise
    summaries = {}
    for condition in config["conditions"]:
        for phase in condition["phases"]:
            selected = []
            for trial in results:
                if trial["condition"] == condition["name"]:
                    for result in trial["phases"]:
                        if result["id"] == phase["id"]:
                            selected.append({**result, "condition": condition["name"], "seed": trial["seed"]})
            summaries.setdefault(condition["name"], {})[phase["id"]] = _summary(selected, phase["analysis"]).get(condition["name"], {})
    output = {"trials": results, "summary": summaries, "analysis": config["analysis"], "resumptions": _extensions(root, config)[0]}
    _write(root / "results.json", output)
    (root / "results").mkdir(exist_ok=True)
    _write(root / "results" / "summary.json", output)
    manifest.update(status="complete_with_failures" if any(t["status"] == "failed" for t in results) else "complete", updated_at=_stamp())
    # Config hashes and receipt chains are authoritative. Results and cursors
    # are derived and may be regenerated after an interrupted report write.
    _write(root / "manifest.json", manifest)
    return Run(root)


def run(experiment, directory="runs", *, workspace=None, original=None):
    from . import __version__
    from .experiments import Workspace

    config = experiment.to_dict()
    _validated_analysis(config)
    destination = workspace.runs if isinstance(workspace, Workspace) else Path(directory) / workspace if workspace is not None else Path(directory)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    root = destination / name
    root.mkdir(parents=True, exist_ok=False)
    resolved = dumps(config)
    (root / "config.toml").write_text(original or resolved, encoding="utf-8")
    (root / "resolved.toml").write_text(resolved, encoding="utf-8")
    _write(root / "resolved.json", config)
    trials = _trial_inventory(config)
    _write(root / "manifest.json", {
        "schema_version": 3, "kind": "run", "status": "pending", "created_at": _stamp(),
        "software": {"self_assembling_games": __version__, "python": sys.version, "platform": platform.platform()},
        "trials": trials,
        "hashes": {name: _hash(root / name) for name in ("config.toml", "resolved.toml", "resolved.json")},
    })
    return _complete(root)


def resume(root, additional_budget=None):
    root = Path(root)
    report = verify(root)
    if not report["valid"]:
        raise ValueError("Run verification failed: " + "; ".join(report["errors"]))
    config = _configuration(root)
    if additional_budget is not None:
        _validate_extension(additional_budget, config)
        entries, _ = _extensions(root, config)
        record = {"at": _stamp(), "additional_budget": additional_budget, "scope": "final_phase", "previous_extension_hash": entries[-1]["extension_hash"] if entries else None}
        record["extension_hash"] = digest(record)
        directory = root / "budget_extensions"
        directory.mkdir(exist_ok=True)
        _write(directory / f"{len(entries):08d}.json", record)
    return _complete(root)
