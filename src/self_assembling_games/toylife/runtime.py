"""Owned canonical chemistry, experience and organism-level selection.

Task scores never update chemical state. Only encoded exposures enter the native
chemistry; every completed native update is a resumable boundary.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np

from . import CONTRACT, interaction
from .apparatus import controller, encode, fingerprint, pack, probabilities, unpack
from .tasks import held_out_cases, static_cases


def _native():
    try:
        import toylife2
    except ImportError as exc:
        raise ImportError(
            "Molecular runs require toyLIFE2 on Python 3.12 or 3.13; install the optional toylife dependency or a local toyLIFE2 checkout"
        ) from exc
    return toylife2


def _provenance(model, data):
    import toylife2

    root = Path(toylife2.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    source = data["interaction"]["provider"]
    extra = None
    if ":" in source:
        obj = interaction(data)
        path = inspect.getsourcefile(type(obj))
        if not path:
            raise ValueError("Custom interactions need inspectable source")
        extra = {
            "provider": source,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    return {
        "contract": CONTRACT,
        "rules": "toylife-canonical-v1",
        "version": toylife2.__version__,
        "source_hash": h.hexdigest(),
        "tables": str(model.tables.digest),
        "table_content": model.tables.content_checksum(),
        "custom_interaction": extra,
    }


class AssayEvaluator:
    """PopulationSession evaluator with pending chemistry and addressed episodes."""

    def __init__(self, native_model, data, seed, *, split="train"):
        self.model, self.data, self.seed = native_model, copy.deepcopy(data), seed
        self.world = interaction(data)
        self.task = getattr(self.world, "task", None)
        self.static = bool(self.task and self.task["static"])
        if split not in ("train", "held_out"):
            raise ValueError("split must be train or held_out")
        if split == "held_out" and (not self.task or self.task["name"] != "ordering"):
            raise ValueError("held_out is defined for ordering tasks")
        self.cases = (
            (
                held_out_cases(self.task)
                if split == "held_out"
                else static_cases(self.task)
            )
            if self.static
            else None
        )
        if self.static and not self.cases:
            raise ValueError("The requested case panel is empty")
        self.apparatus = data["realisation"]["parameters"]["resolved_apparatus"]
        self.controller = controller(
            self.apparatus, data["realisation"]["parameters"]["controller"]
        )
        self.execution = data["execution"]["parameters"]
        self.stochastic = not self.static
        self.context = fingerprint({"model": data, "split": split, "seed": seed})
        self.metrics = {"chemical_updates": 0, "world_ticks": 0}
        self.requests = {}
        self.pending = None
        self.chemistry = None
        self.diagnostics = []
        self.work_limits = None
        self._work_start = dict(self.metrics)

    def set_work_limits(self, limits):
        self.work_limits = limits
        self._work_start = dict(self.metrics)

    def _allow(self, name):
        limit = (
            None
            if self.work_limits is None
            else getattr(self.work_limits, "max_" + name)
        )
        return limit is None or self.metrics[name] - self._work_start[name] < limit

    def take_events(self):
        return []

    def retain_cache_panels(self, panels):
        panels = set(panels)
        self.requests = {
            k: v
            for k, v in self.requests.items()
            if v["panel"] in panels or (self.pending and self.pending["request"] == k)
        }

    def checkpoint_payload(self):
        pending = copy.deepcopy(self.pending)
        if pending is not None and self.chemistry is not None:
            pending["chemistry"] = self.chemistry.payload()
        return dict(
            context=self.context,
            requests=copy.deepcopy(self.requests),
            pending=pending,
            metrics=dict(self.metrics),
            diagnostics=copy.deepcopy(self.diagnostics),
        )

    def restore_payload(self, data):
        if data["context"] != self.context:
            raise ValueError("Molecular evaluator identity mismatch")
        self.requests = copy.deepcopy(data["requests"])
        self.pending = copy.deepcopy(data["pending"])
        self.metrics = dict(data["metrics"])
        self.diagnostics = copy.deepcopy(data["diagnostics"])
        self.chemistry = None

    def _begin(self, key, request, index, unit):
        from toylife2.chemistry import ChemistrySession

        genes = request["genes"][index]
        self.chemistry = ChemistrySession(self.model, genes, self.controller)
        # World draws are common across candidates; actuator draws have a separate
        # stable genotype address. Neither stream depends on batch ordering.
        address = int(fingerprint(genes)[:16], 16)
        world_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, request["panel"], unit, 0])
        )
        action_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, request["panel"], unit, address, 1])
        )
        self.pending = {
            "request": key,
            "index": index,
            "unit": unit,
            "updates": 0,
            "phases": 0,
            "phase_reset_pending": False,
            "score": 0.0,
            "judgements": 0,
            "world": None if self.static else self.world.initial_world(world_rng),
            "world_rng": world_rng.bit_generator.state,
            "action_rng": action_rng.bit_generator.state,
        }

    def _resume_chemistry(self, request):
        if self.chemistry is None:
            from toylife2.chemistry import ChemistrySession

            self.chemistry = ChemistrySession(
                self.model, request["genes"][self.pending["index"]], self.controller
            )
            self.chemistry.restore_payload(self.pending.pop("chemistry"))

    def _record(self, row):
        limit = self.execution["trace_limit"]
        if limit:
            self.diagnostics.append(row)
            del self.diagnostics[:-limit]

    def _unit(self, request):
        self._resume_chemistry(request)
        p = self.pending
        if not self.static and p["phase_reset_pending"]:
            if self.data["realisation"]["parameters"].get(
                "reset_between_phases", False
            ):
                from toylife2.chemistry import ChemistrySession

                self.chemistry = ChemistrySession(
                    self.model, request["genes"][p["index"]], self.controller
                )
            p["phase_reset_pending"] = False
        if self.static:
            case = self.cases[p["unit"]]
            observation, duration, actors = (
                case["observation"],
                self.execution["response_updates"],
                ("cell",),
            )
            legal = self.world.actions
        else:
            stage = self.world.stage(copy.deepcopy(p["world"]))
            if any(actor != "cell" for actor in stage.actors) or len(stage.actors) > 1:
                raise ValueError("A single-pool stage has at most actor cell")
            duration = (
                self.execution["response_updates"]
                if stage.duration is None
                else stage.duration
            )
            if type(duration) is not int or duration < 0:
                raise ValueError(
                    "Canonical stage duration is a nonnegative integer update count"
                )
            actors = stage.actors
            observation = self.world.observations(copy.deepcopy(p["world"]))["cell"]
            legal = (
                self.world.legal_actions(copy.deepcopy(p["world"]), "cell")
                if actors
                else []
            )
        # Only this observation reaches the encoder, never the world or history.
        named = encode(self.apparatus, observation)
        supplied = self.controller.interface.encode(named)
        while p["updates"] < duration:
            if not self._allow("chemical_updates"):
                return False
            before = self.chemistry.update
            self.chemistry.step(supplied)
            if self.chemistry.update != before + 1:
                return False
            p["updates"] += 1
            self.metrics["chemical_updates"] += 1
        if not self._allow("world_ticks"):
            return False
        self.metrics["world_ticks"] += 1
        distribution = (
            probabilities(self.apparatus, self.chemistry.last_readouts, legal)
            if actors
            else None
        )
        if self.static:
            score = float(np.dot(distribution, case["rewards"]))
            self._record(
                {
                    "genotype": request["genes"][p["index"]].tolist(),
                    "case": p["unit"],
                    "observation": observation,
                    "action_probabilities": distribution.tolist(),
                    "score": score,
                    "readouts": {
                        k: np.asarray(v).tolist()
                        for k, v in self.chemistry.last_readouts.items()
                    },
                }
            )
            request["case_scores"][p["index"], p["unit"]] = score
            request["case_probabilities"][p["index"], p["unit"]] = distribution
            request["scores"][p["index"]] += case["probability"] * score
            return True
        wrng = np.random.default_rng()
        wrng.bit_generator.state = p["world_rng"]
        arng = np.random.default_rng()
        arng.bit_generator.state = p["action_rng"]
        actions = {}
        if actors:
            index = int(arng.choice(len(distribution), p=distribution))
            actions["cell"] = self.world.actions[index]
        result = self.world.transition(copy.deepcopy(p["world"]), actions, wrng)
        reward = result.feedback.get("cell", {}).get("score")
        if reward is not None:
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not np.isfinite(reward)
                or not 0 <= reward <= 1
            ):
                raise ValueError(
                    "Task scores must be finite in [0,1]; retain raw returns in info"
                )
            p["score"] += float(reward)
            request["trial_scores"][p["index"], p["unit"], p["judgements"]] = float(
                reward
            )
            p["judgements"] += 1
        self._record(
            {
                "genotype": request["genes"][p["index"]].tolist(),
                "episode": p["unit"],
                "phase": p["phases"],
                "observation": observation,
                "actions": actions,
                "score": reward,
                "action_probabilities": None
                if distribution is None
                else distribution.tolist(),
                "readouts": {
                    k: np.asarray(v).tolist()
                    for k, v in self.chemistry.last_readouts.items()
                },
                "info": result.info,
            }
        )
        p.update(
            world=result.world,
            world_rng=wrng.bit_generator.state,
            action_rng=arng.bit_generator.state,
            updates=0,
            phases=p["phases"] + 1,
            phase_reset_pending=True,
        )
        if result.truncated or (
            p["phases"] >= self.execution["max_episode_phases"]
            and not result.terminated
        ):
            request["truncated"][p["index"]] = True
            return True
        if result.terminated:
            request["scores"][p["index"]] += (
                p["score"] / max(1, p["judgements"]) / self.execution["episodes"]
            )
            return True
        return None

    def evaluate(
        self,
        genotypes,
        candidate_ids=None,
        *,
        panel=0,
        identities=None,
        agent_axis=None,
    ):
        from toylife2.evaluation import EvaluationResult
        from toylife2.genotypes import validate

        if agent_axis is not None:
            raise ValueError("This evaluator owns one pool per candidate")
        genes = validate(genotypes)
        shape = genes.shape[:-1]
        ids = (
            np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
            if candidate_ids is None
            else np.asarray(candidate_ids)
        )
        if ids.shape != shape:
            raise ValueError("candidate_ids must match genotype leading dimensions")
        key = fingerprint({"genes": genes, "panel": int(panel)})
        if self.pending and self.pending["request"] != key:
            raise ValueError(
                "Resume the pending assay before requesting another cohort"
            )
        units = len(self.cases) if self.static else self.execution["episodes"]
        if key not in self.requests:
            count = genes.size // genes.shape[-1]
            actions = len(self.world.actions)
            diagnostic_bytes = (
                count
                * units
                * (
                    8 * (1 + actions)
                    if self.static
                    else 8 * self.execution["max_episode_phases"]
                )
            )
            # Retain one completed request alongside the current pending cohort.
            previous = next(reversed(self.requests), None)
            self.requests = (
                {} if previous is None else {previous: self.requests[previous]}
            )
            retained_bytes = sum(
                v.nbytes
                for r in self.requests.values()
                for v in r.values()
                if isinstance(v, np.ndarray)
            )
            required = diagnostic_bytes + genes.nbytes + count * 32 + retained_bytes
            if required > self.execution["memory_budget_mib"] * 1024 * 1024:
                raise MemoryError(
                    "Molecular assay analysis exceeds the execution memory budget"
                )
            self.model.engine._check_memory(
                required,
                message="Molecular assay analysis exceeds the execution memory budget",
            )
            request = {
                "genes": genes.reshape(-1, genes.shape[-1]).copy(),
                "panel": int(panel),
                "scores": np.zeros(count),
                "truncated": np.zeros(count, bool),
                "index": 0,
                "unit": 0,
            }
            if self.static:
                request.update(
                    case_scores=np.full((count, units), np.nan),
                    case_probabilities=np.full((count, units, actions), np.nan),
                )
            else:
                request["trial_scores"] = np.full(
                    (count, units, self.execution["max_episode_phases"]), np.nan
                )
            self.requests[key] = request
        request = self.requests[key]
        while request["index"] < len(request["genes"]):
            if self.pending is None:
                self._begin(key, request, request["index"], request["unit"])
            done = self._unit(request)
            if done is False:
                break
            if done is None:
                continue
            self.pending = self.chemistry = None
            request["unit"] += 1
            if request["unit"] == units:
                request["index"] += 1
                request["unit"] = 0
        if request["index"] == len(request["genes"]):
            self.requests = {key: request}
        statuses = np.full(len(request["genes"]), "incomplete", dtype="U12")
        statuses[: request["index"]] = "complete"
        statuses[request["truncated"]] = "truncated"
        scores = request["scores"].reshape(shape).copy()
        floor = self.data["adaptation"]["parameters"]["fitness_floor"]
        selection = np.maximum(scores, floor)
        denominator = 1 << 24
        weights = np.rint(selection * denominator).astype(np.int64)
        if floor > 0:
            weights = np.maximum(weights, 1)
        return EvaluationResult(
            scores,
            candidate_ids=ids,
            status=statuses.reshape(shape),
            weights=weights,
            denominator=denominator,
            selection_fitness=selection,
            metrics={
                "chemical_updates": self.metrics["chemical_updates"],
                "world_ticks": self.metrics["world_ticks"],
            },
        )

    def analysis(self, genes, panel):
        request = self.requests[fingerprint({"genes": genes, "panel": int(panel)})]
        if self.static:
            cases = [
                {
                    "observation": copy.deepcopy(case["observation"]),
                    "probability": case["probability"],
                    "rewards": list(case["rewards"]),
                    "scores": request["case_scores"][:, i].tolist(),
                    "action_probabilities": request["case_probabilities"][
                        :, i
                    ].tolist(),
                }
                for i, case in enumerate(self.cases)
            ]
            confusion = None
            if all(
                set(c["rewards"]) <= {0, 1} and sum(c["rewards"]) == 1
                for c in self.cases
            ):
                confusion = np.zeros((len(self.world.actions), len(self.world.actions)))
                for i, case in enumerate(self.cases):
                    confusion[case["rewards"].index(1)] += case[
                        "probability"
                    ] * request["case_probabilities"][:, i].mean(axis=0)
                confusion = {
                    "actions": list(self.world.actions),
                    "target_by_action_probability": confusion.tolist(),
                }
            return {
                "case_metrics": cases,
                "confusion": confusion,
                "episodic_trial_curves": None,
            }
        trials = request["trial_scores"]
        counts = np.isfinite(trials).sum(axis=(0, 1))
        length = int(np.flatnonzero(counts)[-1] + 1) if np.any(counts) else 0
        means = np.divide(
            np.nansum(trials, axis=(0, 1)),
            counts,
            out=np.zeros_like(counts, dtype=float),
            where=counts > 0,
        )
        episodes = [
            [[float(x) for x in row if np.isfinite(x)] for row in candidate]
            for candidate in trials
        ]
        return {
            "case_metrics": None,
            "confusion": None,
            "episodic_trial_curves": {
                "candidate_episode_scores": episodes,
                "mean_by_scored_trial": means[:length].tolist(),
                "observations_by_scored_trial": counts[:length].tolist(),
            },
        }

    def evaluate_population(self, population, *, replicate, generation):
        return self.evaluate(population, panel=generation if self.stochastic else 0)


class MolecularBackend:
    clock = "generations"

    def __init__(self, data, seed, protocol, snapshot=None):
        native = _native()
        from toylife2.specs import ExecutionSpec

        if protocol:
            raise ValueError("Declare molecular settings in the model sections")
        self.data, self.seed, self.protocol = copy.deepcopy(data), seed, dict(protocol)
        self.settings = data["execution"]["parameters"]
        self.model = native.Model(
            execution=ExecutionSpec(
                backend=self.settings["backend"],
                memory_budget_mib=self.settings["memory_budget_mib"],
            )
        )
        self.provenance = _provenance(self.model, data)
        if snapshot is not None and snapshot["provenance"] != self.provenance:
            raise ValueError("Molecular dependency or chemistry identity mismatch")
        self.horizon = (
            data["execution"]["budget"] if snapshot is None else snapshot["horizon"]
        )
        self.evaluator = AssayEvaluator(self.model, data, seed)
        self.spec = self._spec(self.horizon)
        self.session = native.PopulationSession(
            self.model,
            self.spec,
            evaluator=self.evaluator,
            checkpoint=None
            if snapshot is None
            else self._checkpoint(snapshot["population"]),
        )
        self.last_status = "ready" if snapshot is None else snapshot["status"]

    def _spec(self, horizon):
        from toylife2.specs import RunSpec, ModelSpec, EvolutionSpec
        from toylife2.protocols import EpisodeSpec, TaskSpec, ObjectiveSpec

        real = self.data["realisation"]["parameters"]
        adaptation = self.data["adaptation"]["parameters"]
        return RunSpec(
            schema_version=2,
            operation="wright_fisher",
            seed=self.seed,
            genotypes=tuple(real.get("initial_genotypes", ())),
            model=ModelSpec(rules="toylife-canonical-v1", genes=real["genes"]),
            execution=self.model.execution,
            controller=self.evaluator.controller,
            evolution=EvolutionSpec(
                generations=horizon,
                trials=1,
                initial="independent_random",
                **adaptation,
            ),
            task=TaskSpec(
                kind="python",
                entrypoint="self_assembling_games.toylife:interaction",
                config={"adapter_contract": CONTRACT, "model": self.data},
            ),
            episode=EpisodeSpec(
                panel="generation" if self.evaluator.stochastic else "fixed"
            ),
            objective=ObjectiveSpec(kind="bounded_reward"),
        )

    @staticmethod
    def _checkpoint(value):
        from toylife2.sessions import Checkpoint

        value = unpack(value)
        return Checkpoint(
            value["kind"], value["identity"], value["state"], value["arrays"]
        )

    @property
    def position(self):
        # A generation is completed only when its population has been assessed.
        # Keep a resource-limited final assessment resumable in the common runner.
        return min(int(self.session.cursors[0]), max(0, int(self.session.observed[0])))

    @property
    def status(self):
        return self.last_status

    def advance(self, amount):
        from toylife2 import PopulationSession, WorkLimits

        if type(amount) is not int or amount < 0:
            raise ValueError("advance requires a nonnegative generation count")
        target = self.position + amount
        if target > self.horizon:
            self.spec, checkpoint = self.session.extension(target)
            self.horizon = target
        else:
            checkpoint = self.session.checkpoint()
        limits = WorkLimits(
            max_evolution_updates=amount,
            max_chemical_updates=self.settings.get("max_chemical_updates"),
            max_world_ticks=self.settings.get("max_world_ticks"),
        )
        self.session = PopulationSession(
            self.model,
            self.spec,
            evaluator=self.evaluator,
            limits=limits,
            checkpoint=checkpoint,
        )
        report = self.session.observe()
        while self.position < target and report.status not in (
            "partial",
            "failed",
            "complete",
        ):
            report = self.session.step()
        self.last_status = report.status
        if (
            report.status == "partial"
            and report.reason == "max_evolution_updates"
            and self.position == target
        ):
            self.last_status = "ready"
        return self

    def snapshot(self):
        cp = self.session.checkpoint()
        return {
            "seed": self.seed,
            "protocol": self.protocol,
            "horizon": self.horizon,
            "provenance": self.provenance,
            "status": self.last_status,
            "population": pack(
                {
                    "kind": cp.kind,
                    "identity": cp.identity,
                    "state": cp.state,
                    "arrays": cp.arrays,
                }
            ),
        }

    def inspect(self):
        interface = self.evaluator.controller.interface
        return {
            "contract": CONTRACT,
            "position": self.position,
            "status": self.status,
            "organisation": self.data["organisation"],
            "genotypes": self.session.populations.tolist(),
            "task": self.evaluator.task,
            "apparatus": self.evaluator.apparatus,
            "endpoint_information": interface.endpoint_information(self.model.tables),
            "provenance": self.provenance,
            "work": dict(self.evaluator.metrics),
            "pending": None
            if self.evaluator.pending is None
            else {
                k: v
                for k, v in self.evaluator.pending.items()
                if k not in ("chemistry", "world_rng", "action_rng")
            },
            "trace": copy.deepcopy(self.evaluator.diagnostics),
        }

    def evaluate(self, *, split="train", panel=0, genotypes=None, episodes=None):
        from toylife2.genotypes import encode as encode_genes, decode

        genes = (
            self.session.populations[0]
            if genotypes is None
            else encode_genes(genotypes)
        )
        if type(panel) is not int or panel < 0:
            raise ValueError("panel must be a nonnegative integer")
        evaluation_data = copy.deepcopy(self.data)
        if episodes is not None:
            if type(episodes) is not int or episodes < 1:
                raise ValueError("episodes must be a positive integer")
            if self.evaluator.static:
                raise ValueError("episodes applies to episodic tasks only")
            evaluation_data["execution"]["parameters"]["episodes"] = episodes
        evaluator = AssayEvaluator(self.model, evaluation_data, self.seed, split=split)
        result = evaluator.evaluate(genes, panel=panel)
        if result is None or not result.complete:
            return {
                "status": "truncated"
                if np.any(result.status == "truncated")
                else "incomplete",
                "candidate_statuses": result.status.reshape(-1).tolist(),
                "status_detail": "task_phase_limit"
                if np.any(result.status == "truncated")
                else "pending_work",
                "position": self.position,
                "work": dict(evaluator.metrics),
            }
        scores = result.scores.reshape(-1)
        best = int(np.argmax(scores))
        return {
            "status": "complete",
            "position": self.position,
            "split": split,
            "panel": int(panel),
            "sampling": {
                "kind": "exact_cases"
                if evaluator.static
                else "shared_world_genotype_addressed_actions",
                "episodes": None
                if evaluator.static
                else evaluator.execution["episodes"],
                "independent_population_lifetimes": False,
            },
            "success_probability": float(scores.mean()),
            "best_score": float(scores[best]),
            "scores": scores.tolist(),
            "selection_fitness": result.selection_fitness.reshape(-1).tolist(),
            "selection_weights": result.weights.reshape(-1).tolist(),
            "selection_denominator": result.denominator,
            "evaluation_overlap": bool(
                split == "held_out"
                and evaluator.task["parameters"]["train_pairs"] == "all"
            ),
            **evaluator.analysis(genes, panel),
            "best_genotype": decode(genes.reshape(-1, genes.shape[-1])[best]),
            "work": dict(evaluator.metrics),
        }
