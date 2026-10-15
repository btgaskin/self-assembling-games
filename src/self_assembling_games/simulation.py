"""Lifecycle, immutable exported state, and version-pinned resumability."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .core import Outcome, Runtime
from .metrics import Metrics
from .registry import Profile, load_profile
from .specification import ModelSpecification
from .model import ModelInstance

SCHEMA = 1


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@lru_cache(maxsize=1)
def _cached_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    package = Path(__file__).parent
    contents = [
        (str(p.relative_to(package)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(package.rglob("*"))
        if p.suffix in {".py", ".json"}
    ]
    return {
        "package_version": "0.2.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rng": "PCG64",
        "repository_revision": revision,
        "source_hash": digest(contents),
    }


def provenance() -> dict[str, Any]:
    """Return owned metadata so solver-specific additions stay local."""
    return copy.deepcopy(_cached_provenance())


def read_snapshot(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.json"
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    if digest(value["profile"]) != value["configuration_hash"]:
        raise ValueError("Checkpoint configuration hash mismatch")
    return value


class Simulation:
    def __init__(
        self, profile: Profile | ModelSpecification, seed: int = 0, *, protocol=None
    ):
        specification = (
            profile
            if isinstance(profile, ModelSpecification)
            else ModelSpecification.from_profile(profile)
        )
        if specification.realisation_id != "barrett.conditional":
            raise ValueError("Simulation needs a Barrett conditional realisation")
        self.profile = specification.to_profile(protocol)
        profile = self.profile
        self.seed = seed
        initial = specification.to_dict()["realisation"]["initial_conditions"]
        self.runtime = Runtime(
            seed,
            initial_weight=initial["table_weight"],
            initial_reference=initial["reference"],
        )
        self.model_instance = ModelInstance(
            specification,
            specification.game_definition,
            self.runtime.organisation,
            self.runtime,
        )
        self.history: list[dict[str, Any]] = []
        self.protocol = specification.execution_parameters(protocol)
        self.metrics = Metrics(self.protocol["window"])
        self.metadata = provenance()
        self.status = "ready"
        self.model_instance.initialize()
        self.configuration_hash = digest(self.profile.to_dict())
        if profile.learner["rule"] == "R3":
            self.history.append(
                {
                    "phase_schedule": "alternating_reward_punishment",
                    "phase_length": profile.learner["phase_length"],
                }
            )

    def step(self) -> Outcome:
        self.runtime.begin()
        outcome = self.model_instance.encounter()
        if outcome.owner_returns is not None:
            outcome.returns_for(self.runtime.owners.values())
        self.metrics.add(outcome)
        self.runtime.emit(
            "outcome",
            success=outcome.success,
            base_reward=outcome.base_reward,
            cost=outcome.cost,
            payoff=outcome.payoff,
            status=outcome.status,
            terms=outcome.terms,
            exchanges=outcome.exchanges,
            details=outcome.details,
            **({"owner_returns": outcome.owner_returns} if outcome.owner_returns is not None else {}),
        )
        self.runtime.clocks["play"] += 1
        self.status = outcome.status
        return outcome

    def run(
        self,
        *,
        plays: int | None = None,
        output: str | Path | None = None,
        checkpoint_every: int = 0,
        trace: str | Path | None = None,
    ) -> dict[str, Any]:
        count = self.protocol["plays"] if plays is None else plays
        if not isinstance(count, int) or count < 0:
            raise ValueError("plays must be nonnegative")
        if checkpoint_every < 0:
            raise ValueError("checkpoint_every must be nonnegative")
        if trace:
            Path(trace).parent.mkdir(parents=True, exist_ok=True)
        handle = Path(trace).open("a") if trace else None
        if handle:
            self.runtime.sink = lambda event: handle.write(
                json.dumps(event, allow_nan=False) + "\n"
            )
            self.runtime.emit("trace_start", snapshot=self.snapshot())
        try:
            for _ in range(count):
                outcome = self.step()
                if (
                    output
                    and checkpoint_every
                    and self.runtime.clocks["play"] % checkpoint_every == 0
                ):
                    self.checkpoint(output)
                if outcome.status == "resource_limit":
                    break
        finally:
            if handle:
                handle.close()
            self.runtime.sink = None
        if output:
            self.checkpoint(output)
        return self.metrics.summary()

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "schema_version": SCHEMA,
                "profile": self.profile.to_dict(),
                "configuration_hash": self.configuration_hash,
                "seed": self.seed,
                "runtime": self.runtime.state(),
                "world": self.world,
                "interaction": self.interaction,
                "metrics": self.metrics.state(),
                "lineage": self.lineage,
                "history": self.history,
                "metadata": self.metadata,
                "status": self.status,
                "model_specification": self.model_instance.specification.to_dict(),
                "model_specification_hash": digest(
                    self.model_instance.specification.to_dict()
                ),
                "model_protocol": self.protocol,
            }
        )

    def checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".json":
            path = path / "checkpoint.json"
        atomic_json(path, self.snapshot())
        return path

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> Simulation:
        if snapshot.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported checkpoint schema")
        current = provenance()
        for field in ["package_version", "numpy", "source_hash"]:
            if snapshot["metadata"].get(field) != current[field]:
                raise ValueError(
                    f"Checkpoint {field} differs from this runtime; use the pinned implementation"
                )
        simulation = cls.__new__(cls)
        simulation.profile = load_profile(snapshot["profile"])
        if digest(simulation.profile.to_dict()) != snapshot["configuration_hash"]:
            raise ValueError("Checkpoint configuration hash mismatch")
        simulation.configuration_hash = snapshot["configuration_hash"]
        simulation.seed = snapshot["seed"]
        simulation.runtime = Runtime.restore(snapshot["runtime"])
        for name in [
            "world",
            "interaction",
            "lineage",
            "history",
            "metadata",
            "status",
        ]:
            setattr(simulation, name, copy.deepcopy(snapshot[name]))
        simulation.metrics = Metrics.restore(snapshot["metrics"])
        declaration = snapshot.get("model_specification")
        if declaration is not None and digest(declaration) != snapshot.get(
            "model_specification_hash"
        ):
            raise ValueError("Checkpoint model specification hash mismatch")
        specification = ModelSpecification.from_profile(
            simulation.profile
        ).restore_declaration(declaration)
        simulation.model_instance = ModelInstance(
            specification,
            specification.game_definition,
            simulation.runtime.organisation,
            simulation.runtime,
        )
        simulation.protocol = specification.execution_parameters()
        if (
            "model_protocol" in snapshot
            and simulation.protocol != snapshot["model_protocol"]
        ):
            raise ValueError("Checkpoint model protocol differs")
        return simulation

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        self._runtime = value
        if hasattr(self, "model_instance"):
            previous = self.model_instance
            self.model_instance = ModelInstance(
                previous.specification, previous.game, value.organisation, value
            )

    @property
    def world(self):
        return self.runtime.world

    @world.setter
    def world(self, value):
        self.runtime.world = value

    @property
    def interaction(self):
        return self.runtime.interaction

    @interaction.setter
    def interaction(self, value):
        self.runtime.interaction = value

    @property
    def lineage(self):
        return self.runtime.lineage

    @lineage.setter
    def lineage(self, value):
        self.runtime.lineage = value

    def inspect(self) -> dict[str, Any]:
        return {
            "model_instance": self.model_instance.inspect(),
            "profile": self.profile.to_dict(),
            "clocks": dict(self.runtime.clocks),
            "world": copy.deepcopy(self.world),
            "interaction": copy.deepcopy(self.interaction),
            "owners": dict(self.runtime.owners),
            "lineage": copy.deepcopy(self.lineage),
            "module_roles": dict(self.runtime.module_roles),
            "metrics": self.metrics.summary(),
            "status": self.status,
            "tables": {
                identity: {
                    **table.to_dict(),
                    "response_probabilities": table.probabilities().tolist(),
                    "resampling_probabilities": table.probabilities(
                        response=False
                    ).tolist(),
                }
                for identity, table in self.runtime.tables.items()
            },
        }

    def set_policy(
        self, module: str, context: Any, option: Any, *, frozen: bool = True
    ) -> None:
        """Construct a named validation fixture or deliberate fixed policy."""
        table = self.runtime.tables[self.runtime.table_id(module, context)]
        table.index(option)
        table.current = option
        table.frozen = frozen
        self.history.append(
            {
                "play": self.runtime.clocks["play"],
                "fixture": "set_policy",
                "module": module,
                "context": context,
                "option": option,
            }
        )


def build(
    profile: str | Profile | dict[str, Any] | ModelSpecification,
    seed: int = 0,
    *,
    protocol=None,
):
    from .adaptive_runtime import build_adaptive

    specification = (
        profile
        if isinstance(profile, ModelSpecification)
        else ModelSpecification.from_profile(profile)
    )
    if specification.realisation_id != "barrett.conditional":
        return build_adaptive(specification, seed, protocol=protocol)
    return Simulation(specification, seed, protocol=protocol)


def resume(path: str | Path) -> Simulation:
    snapshot = read_snapshot(path)
    if snapshot.get("engine") == "adaptive":
        from .adaptive_runtime import restore_adaptive

        return restore_adaptive(snapshot)
    return Simulation.restore(snapshot)


def run_ensemble(
    profile: str | Profile | dict[str, Any],
    seeds: list[int],
    *,
    output: str | Path,
    threshold: float | None = None,
    plays: int | None = None,
    workers: int = 1,
    duration: float | None = None,
    metric: str | None = None,
    generations: int | None = None,
) -> dict[str, Any]:
    from .adaptive_runtime import is_adaptive_profile

    if is_adaptive_profile(profile):
        from .adaptive_experiments import adaptive_ensemble

        return adaptive_ensemble(
            profile,
            seeds,
            output=output,
            threshold=threshold,
            plays=plays,
            duration=duration,
            metric=metric,
            workers=workers,
            generations=generations,
        )
    if (
        duration is not None
        or generations is not None
        or metric not in {None, "cumulative_accuracy"}
    ):
        raise ValueError(
            "Barrett encounter ensembles use plays and cumulative_accuracy"
        )
    from .metrics import wilson

    resolved = load_profile(profile)
    root = Path(output)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be unique and nonempty")
    duration = resolved.plays if plays is None else plays
    if not isinstance(duration, int) or duration <= 0:
        raise ValueError("Ensemble plays must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Accuracy threshold is between zero and one")
    manifest = {
        "profile": resolved.to_dict(),
        "configuration_hash": digest(resolved.to_dict()),
        "seeds": seeds,
        "plays": duration,
        "threshold": threshold,
        "metric": "cumulative_accuracy",
    }
    manifest_path = root / "ensemble.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing ensemble has a different protocol")
    atomic_json(manifest_path, manifest)
    tasks = [
        (resolved.to_dict(), seed, duration, str(root / f"seed-{seed}"))
        for seed in seeds
    ]
    if workers == 1:
        results = [_ensemble_run(task) for task in tasks]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(_ensemble_run, tasks))
    accuracies = [
        r["cumulative_accuracy"]
        for r in results
        if r["cumulative_accuracy"] is not None
    ]
    report = {
        "protocol": manifest,
        "runs": results,
        "mean_cumulative_accuracy": float(np.mean(accuracies)) if accuracies else None,
        "incomplete_runs": sum(r["completed"] != duration for r in results),
    }
    if accuracies:
        boot_rng = np.random.Generator(np.random.PCG64(0))
        bootstrap = np.mean(
            boot_rng.choice(accuracies, (2000, len(accuracies))), axis=1
        )
        report["mean_accuracy_bootstrap_95"] = np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist()
        report["mean_interval_protocol"] = {
            "method": "percentile_bootstrap",
            "resamples": 2000,
            "seed": 0,
        }
    if threshold is not None:
        passed = sum(
            r["completed"] == duration and r["cumulative_accuracy"] > threshold
            for r in results
        )
        report.update(
            {
                "threshold_runs": passed,
                "threshold_fraction": passed / len(results),
                "threshold_wilson_95": wilson(passed, len(results)),
            }
        )
    atomic_json(root / "report.json", report)
    return report


def _ensemble_run(task: tuple[dict[str, Any], int, int, str]) -> dict[str, Any]:
    profile, seed, duration, location = task
    destination = Path(location)
    checkpoint = destination / "checkpoint.json"
    sim = resume(checkpoint) if checkpoint.exists() else build(profile, seed)
    if sim.seed != seed or sim.configuration_hash != digest(profile):
        raise ValueError("Ensemble checkpoint does not match the registered protocol")
    remaining = duration - sim.runtime.clocks["play"]
    if remaining < 0:
        raise ValueError("Checkpoint exceeds registered duration")
    if sim.status != "resource_limit":
        sim.run(plays=remaining, output=destination)
    return {"seed": seed, **sim.metrics.summary()}


def benchmark(
    profile: str,
    plays: int | None = None,
    seed: int = 0,
    *,
    duration: float | None = None,
    generations: int | None = None,
) -> dict[str, Any]:
    sim = build(profile, seed)
    from .evolution import EvolutionSimulation
    from .finite_learning import ReinforcementSimulation
    from .multilocus import MultilocusSimulation

    if isinstance(sim, MultilocusSimulation):
        if plays is not None or duration is not None:
            raise ValueError("Multilocus benchmarks use generations")
        start = time.perf_counter()
        sim.run(generations=generations)
        return {
            "profile": profile,
            "clock": "generations",
            "generations": sim.generation,
            "elapsed_seconds": time.perf_counter() - start,
            "metadata": sim.metadata,
        }
    if generations is not None:
        raise ValueError("generations belongs to multilocus generation methods")

    if isinstance(sim, EvolutionSimulation):
        if plays is not None:
            raise ValueError("Evolution benchmarks use duration, not plays")
        start = time.perf_counter()
        sim.run(duration=duration)
        return {
            "profile": profile,
            "clock": "evolutionary_time",
            "evolutionary_time": sim.time,
            "elapsed_seconds": time.perf_counter() - start,
            "metadata": sim.metadata,
            "solver": sim.protocol["solver"],
        }
    if duration is not None:
        raise ValueError("Reinforcement benchmarks use plays")
    plays = 10_000 if plays is None else plays
    start = time.perf_counter()
    sim.run(plays=plays)
    elapsed = time.perf_counter() - start
    if isinstance(sim, ReinforcementSimulation):
        return {
            "profile": profile,
            "clock": "plays",
            "plays": plays,
            "elapsed_seconds": elapsed,
            "plays_per_second": plays / elapsed,
            "metadata": sim.metadata,
            "adaptive_vectors": len(sim.weights),
        }
    return {
        "profile": profile,
        "plays": plays,
        "elapsed_seconds": elapsed,
        "plays_per_second": plays / elapsed,
        "metadata": sim.metadata,
        "table_count": len(sim.runtime.tables),
        "retained_events": len(sim.runtime.events),
    }
