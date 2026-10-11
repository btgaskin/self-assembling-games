"""Resumable independent-seed experiments with method-specific measurements."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite_profiles import AdaptiveProfile, load_adaptive_profile


def adaptive_ensemble(
    profile: str | dict[str, Any] | AdaptiveProfile,
    seeds: list[int],
    *,
    output: str | Path,
    duration: float | None = None,
    plays: int | None = None,
    metric: str | None = None,
    threshold: float | None = None,
    workers: int = 1,
    generations: int | None = None,
) -> dict[str, Any]:
    import json
    import math
    from .simulation import atomic_json, digest
    from .metrics import wilson

    p = load_adaptive_profile(profile)
    method_class = adaptive_method(p.method).method_class
    evolution = method_class == "population_evolution"
    discrete = method_class == "population_generations"
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique and nonempty")
    if not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("Accuracy threshold must be in [0,1]")
    if (
        (evolution and (plays is not None or generations is not None))
        or (discrete and (plays is not None or duration is not None))
        or (
            method_class == "reinforcement"
            and (duration is not None or generations is not None)
        )
    ):
        raise ValueError("Ensemble duration must use the adaptive method's clock")
    horizon = (
        (p.protocol.get("generations", 100) if generations is None else generations)
        if discrete
        else (p.protocol.get("duration", 50.0) if duration is None else duration)
        if evolution
        else (p.protocol.get("plays", 10_000) if plays is None else plays)
    )
    if (
        not isinstance(horizon, (int, float))
        or not math.isfinite(horizon)
        or horizon <= 0
    ):
        raise ValueError("Ensemble horizon must be finite and positive")
    if not evolution and (not isinstance(horizon, int) or isinstance(horizon, bool)):
        raise ValueError("Ensemble plays/generations must be an integer")
    metric = metric or (
        "final_policy_success_probability"
        if evolution or discrete
        else "cumulative_accuracy"
    )
    allowed = {
        "final_policy_success_probability",
        "generation_average_expected_success"
        if discrete
        else "time_average_expected_success"
        if evolution
        else "cumulative_accuracy",
    }
    if metric not in allowed:
        raise ValueError("Metric does not belong to this adaptive method")
    root = Path(output)
    manifest = {
        "profile": p.to_dict(),
        "configuration_hash": digest(p.to_dict()),
        "seeds": list(seeds),
        "horizon": horizon,
        "clock": "generations"
        if discrete
        else "evolutionary_time"
        if evolution
        else "plays",
        "metric": metric,
        "threshold": threshold,
        "threshold_comparison": "strictly_greater",
    }
    path = root / "ensemble.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing ensemble has a different protocol")
    atomic_json(path, manifest)
    tasks = [
        (p.to_dict(), seed, horizon, str(root / f"seed-{seed}"), metric)
        for seed in seeds
    ]
    if workers == 1:
        results = [_adaptive_run(t) for t in tasks]
    else:
        if not p.model.startswith(("SK-", "ML-")) and "profile" not in p.parameters:
            raise ValueError(
                "Custom game factories need serial ensembles or worker-side registration"
            )
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(_adaptive_run, tasks))
    values = [r["measurement"] for r in results if r["complete"]]
    report = {
        "protocol": manifest,
        "runs": results,
        "complete_runs": len(values),
        "incomplete_runs": len(results) - len(values),
        "mean_measurement": float(np.mean(values)) if values else None,
    }
    if values:
        rng = np.random.Generator(np.random.PCG64(0))
        bootstrap = np.mean(rng.choice(values, (2000, len(values))), axis=1)
        report["mean_bootstrap_95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        report["interval_protocol"] = {
            "method": "percentile_bootstrap",
            "resamples": 2000,
            "seed": 0,
        }
    if threshold is not None:
        passed = sum(r["complete"] and r["measurement"] > threshold for r in results)
        report.update(
            threshold_runs=passed,
            threshold_fraction=passed / len(results),
            threshold_wilson_95=wilson(passed, len(results)),
            threshold_denominator="all_registered_seeds",
        )
    atomic_json(root / "report.json", report)
    return copy.deepcopy(report)


def _adaptive_run(task):
    from .adaptive_runtime import build_adaptive
    from .simulation import digest, resume
    from .evolution import (
        EvolutionSimulation,
        EvolutionResourceError,
        IntegrationDomainError,
    )

    profile, seed, horizon, destination, metric = task
    destination = Path(destination)
    checkpoint = destination / "checkpoint.json"
    sim = resume(checkpoint) if checkpoint.exists() else build_adaptive(profile, seed)
    if sim.seed != seed or sim.configuration_hash != digest(profile):
        raise ValueError("Ensemble checkpoint does not match the protocol")
    evolution = isinstance(sim, EvolutionSimulation)
    discrete = sim.method.method_class == "population_generations"
    reached = sim.generation if discrete else sim.time if evolution else sim.plays
    remaining = horizon - reached
    if remaining < 0:
        raise ValueError("Checkpoint exceeds ensemble horizon")
    if remaining > 0 and sim.status not in {"resource_limit", "integration_failed"}:
        if discrete:
            sim.run(generations=remaining, output=destination)
        elif evolution:
            try:
                sim.run(duration=remaining, output=destination)
            except (EvolutionResourceError, IntegrationDomainError):
                pass  # The checkpoint and status retain this incomplete run.
        else:
            sim.run(plays=remaining, output=destination)
    summary = sim.summary()
    if metric == "final_policy_success_probability":
        measurement = sim.evaluate()["success_probability"]
    else:
        measurement = summary[metric]
    return {
        "seed": seed,
        "complete": (
            sim.generation if discrete else sim.time if evolution else sim.plays
        )
        == horizon,
        "measurement": measurement,
        "summary": summary,
        "fidelity": copy.deepcopy(sim.fidelity),
    }
