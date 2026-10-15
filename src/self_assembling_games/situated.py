"""Sampled, staged interactions with private views and restorable local policies.

A stage chooses one batch of actions from one pre-transition world. Actor names
identify participants; the protocol assigns no communicative roles.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Stage:
    """The actors choosing simultaneously at a named decision stage."""

    id: str
    actors: tuple[str, ...]


@dataclass
class StageResult:
    """Explicit world transition, actor feedback, and encounter end conditions."""

    world: dict[str, Any]
    feedback: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Interaction(Protocol):
    """A sampled world: all retained mutable state belongs in ``world``.

    Factories receive the interaction parameter dictionary. Methods receive
    copies, allowing the runtime to retain its pre-state unchanged while actors
    choose. ``legal_actions`` may be an indexable lazy space such as ``range``.
    The runtime never constructs joint policies or complete contingent plans.
    """

    actors: Sequence[str]

    def initial_world(self, rng: np.random.Generator) -> dict[str, Any]: ...
    def stage(self, world: dict[str, Any]) -> Stage: ...
    def observations(self, world: dict[str, Any]) -> Mapping[str, Any]: ...
    def legal_actions(self, world: dict[str, Any], actor: str) -> Sequence[Any]: ...
    def transition(
        self, world: dict[str, Any], actions: dict[str, Any], rng: np.random.Generator
    ) -> StageResult: ...


@runtime_checkable
class LocalController(Protocol):
    """An action rule given only its actor's observable view and private memory."""

    def choose(
        self, view: dict[str, Any], legal_actions: Sequence[Any],
        state: Any, rng: np.random.Generator,
    ) -> tuple[Any, Any]: ...


@runtime_checkable
class LocalAdaptation(Protocol):
    """A feedback update with independently retained adaptive state."""

    def update(
        self, event: dict[str, Any], controller_state: Any,
        adaptation_state: Any, rng: np.random.Generator,
    ) -> tuple[Any, Any]: ...


def _json_copy(value: Any) -> Any:
    """Validate checkpointability and detach mutable data at public boundaries."""
    return json.loads(json.dumps(value, allow_nan=False))


def _factory(identity: str, parameters: dict[str, Any]):
    if not isinstance(identity, str) or ":" not in identity:
        raise ValueError("Custom provider must be an importable module:factory identity")
    module, name = identity.split(":", 1)
    factory = importlib.import_module(module)
    for part in name.split("."):
        factory = getattr(factory, part)
    return factory(copy.deepcopy(parameters))


def _provider_fingerprints(model):
    """Pin external component code as well as each importable factory identity."""
    result = {}
    for name in ("interaction", "realisation", "adaptation"):
        identity = model[name]["provider"]
        if ":" not in identity:
            continue
        module = importlib.import_module(identity.split(":", 1)[0])
        path = getattr(module, "__file__", None)
        if not path:
            raise ValueError("Custom providers require a reproducible importable module file")
        # Source files are stable across .pyc cache refreshes and new processes.
        source = importlib.util.source_from_cache(path) if str(path).endswith(".pyc") else path
        result[name] = {"identity": identity, "module": module.__name__,
                        "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    return result


class RandomController:
    def choose(self, view, legal_actions, state, rng):
        # Index one sampled action; large lazy spaces stay lazy.
        if len(legal_actions) == 0:
            raise ValueError("An active actor requires at least one legal action")
        return legal_actions[int(rng.integers(len(legal_actions)))], state


class ScriptedController:
    def __init__(self, parameters):
        self.scripts = copy.deepcopy(parameters.get("scripts", {}))
        self.fallback = parameters.get("fallback", "wait")

    def choose(self, view, legal_actions, state, rng):
        state = {} if state is None else copy.deepcopy(state)
        index = state.get("index", 0)
        script = self.scripts.get(view["actor"], [])
        action = script[index] if index < len(script) else self.fallback
        state["index"] = index + 1
        return action, state


class NoAdaptation:
    def update(self, event, controller_state, adaptation_state, rng):
        return controller_state, adaptation_state


class RewardTraceAdaptation:
    """Retain feedback counts independently of a controller's policy memory."""

    def update(self, event, controller_state, adaptation_state, rng):
        state = {"updates": 0, "reward_sum": 0.0} if adaptation_state is None else adaptation_state
        state["updates"] += 1
        feedback = event.get("feedback")
        if isinstance(feedback, dict):
            state["reward_sum"] += float(feedback.get("reward", 0.0))
        return controller_state, state


class ECGInteraction:
    """Discrete-time constructed colour-board coordination variant.

    Each board's cells are iid uniform draws, jointly conditioned on the boards
    sharing at least one colour. All six actions are ordinary world actions.
    """

    actors = ("A", "B")
    actions = ("up", "down", "left", "right", "wait", "finish")

    def __init__(self, parameters):
        self.size = parameters.get("size", 2)
        self.colours = tuple(parameters.get("colours", ("red", "green", "blue", "yellow")))
        self.tick_cap = parameters.get("tick_cap", 64)
        if type(self.size) is not int or self.size < 1:
            raise ValueError("ECG size must be a positive integer")
        if not self.colours or any(not isinstance(c, str) for c in self.colours):
            raise ValueError("ECG colours must be a nonempty sequence of strings")
        if len(set(self.colours)) != len(self.colours):
            raise ValueError("ECG colour labels must be distinct")
        if type(self.tick_cap) is not int or self.tick_cap < 1:
            raise ValueError("ECG tick_cap must be a positive integer")

    def initial_world(self, rng):
        cells = self.size ** 2
        while True:
            boards = {
                actor: [self.colours[int(rng.integers(len(self.colours)))] for _ in range(cells)]
                for actor in self.actors
            }
            if set(boards["A"]) & set(boards["B"]):
                break
        return {
            "boards": boards,
            "positions": {a: int(rng.integers(cells)) for a in self.actors},
            "finished": {a: False for a in self.actors}, "tick": 0,
        }

    def stage(self, world):
        return Stage("tick", tuple(a for a in self.actors if not world["finished"][a]))

    def observations(self, world):
        return {
            a: {"board": copy.deepcopy(world["boards"][a]), "size": self.size,
                "positions": copy.deepcopy(world["positions"]),
                "finished": copy.deepcopy(world["finished"]), "tick": world["tick"]}
            for a in self.actors
        }

    def legal_actions(self, world, actor):
        return self.actions if not world["finished"][actor] else ()

    def transition(self, world, actions, rng):
        for actor, action in actions.items():
            if world["finished"][actor]:
                raise ValueError("A finished actor cannot act")
            row, col = divmod(world["positions"][actor], self.size)
            dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1),
                      "right": (0, 1), "wait": (0, 0), "finish": (0, 0)}[action]
            if 0 <= row + dr < self.size and 0 <= col + dc < self.size:
                world["positions"][actor] = (row + dr) * self.size + col + dc
            if action == "finish":
                world["finished"][actor] = True
        world["tick"] += 1
        terminated = all(world["finished"].values())
        truncated = not terminated and world["tick"] >= self.tick_cap
        feedback: dict[str, Any] = {a: {} for a in self.actors}
        info: dict[str, Any] = {"tick": world["tick"]}
        if terminated:
            chosen = {a: world["boards"][a][world["positions"][a]] for a in self.actors}
            reward = int(chosen["A"] == chosen["B"])
            feedback = {
                a: {"reward": reward, "boards": copy.deepcopy(world["boards"]),
                    "chosen_colours": copy.deepcopy(chosen)} for a in self.actors
            }
            info["reward"] = reward
        elif truncated:
            info["reason"] = "tick_cap"
        return StageResult(world, feedback, terminated, truncated, info)


def ecg_factory(parameters):
    """Importable fixture identity useful in explicit custom configurations."""
    return ECGInteraction(parameters)


def random_controller(parameters):
    return RandomController()


def scripted_controller(parameters):
    return ScriptedController(parameters)


def no_adaptation(parameters):
    return NoAdaptation()


ECG_MODEL = {
    "id": "ecg.demo",
    "interaction": {"provider": "situated", "game": "ecg", "parameters": {
        "size": 2, "colours": ["red", "green", "blue", "yellow"], "tick_cap": 64}},
    "organisation": {},
    "realisation": {"provider": "random", "parameters": {}},
    "adaptation": {"provider": "none", "parameters": {}},
    "source": {"kind": "constructed_variant", "identifiers": [
        "doi:10.1016/j.cognition.2009.08.009"], "assumptions": [
            "Discrete synchronous ticks replace continuous interaction.",
            "Private iid colour boards use the declared dimensions and palette, conditioned on sharing a colour.",
            "Random or explicit scripted local controllers are demonstration policies."
        ], "questions": ["What local learning mechanism could acquire coordination?"]},
    "execution": {"clock": "encounters", "budget": 100, "parameters": {}},
    "resources": {},
    "capabilities": ["sampled_interaction", "private_observations", "simultaneous_actions",
                     "staged_checkpoint", "local_controller_memory"],
    "runtime": "situated",
}


def resolve_model(identifier, overrides=None) -> dict[str, Any]:
    if isinstance(identifier, str):
        if identifier.lower() not in {"ecg", "ecg-inspired", "ecg.demo", "ecg.scripted"}:
            raise ValueError(f"Unknown situated model {identifier}")
        model = copy.deepcopy(ECG_MODEL)
        if identifier.lower() == "ecg.scripted":
            model["id"] = "ecg.scripted"
            model["realisation"] = {"provider": "scripted", "parameters": {
                "scripts": {"A": ["wait", "finish"], "B": ["right", "finish"]},
                "fallback": "finish"}}
    elif isinstance(identifier, dict):
        model = copy.deepcopy(identifier)
    elif hasattr(identifier, "to_dict"):
        model = copy.deepcopy(identifier.to_dict())
    else:
        raise TypeError("A situated model must be an identifier or model dictionary")
    if overrides:
        def merge(destination, patch):
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(destination.get(key), dict):
                    merge(destination[key], value)
                else:
                    destination[key] = copy.deepcopy(value)
        merge(model, overrides)
    if model.get("runtime", "situated") != "situated":
        raise ValueError("Situated model requires runtime='situated'")
    model["runtime"] = "situated"
    for name in ("interaction", "organisation", "realisation", "adaptation", "source", "execution"):
        if not isinstance(model.get(name), dict):
            raise ValueError(f"Situated model requires a {name} table")
    for name, builtins in (("interaction", {"situated"}),
                           ("realisation", {"random", "scripted"}),
                           ("adaptation", {"none", "reward_trace"})):
        spec = model[name]
        provider = spec.get("provider")
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"Situated {name} requires a provider identity")
        if provider not in builtins and (":" not in provider or not all(provider.split(":", 1))):
            raise ValueError(f"Situated {name} provider must be a built-in or module:factory")
        if not isinstance(spec.get("parameters", {}), dict):
            raise ValueError(f"Situated {name} parameters must be a table")
    if model["interaction"]["provider"] == "situated":
        if model["interaction"].get("game") != "ecg":
            raise ValueError("Unknown situated interaction game")
        ECGInteraction(model["interaction"].get("parameters", {}))
    if model["execution"].get("clock") != "encounters":
        raise ValueError("Situated interactions use clock='encounters'")
    budget = model["execution"].get("budget")
    if type(budget) is not int or budget < 0:
        raise ValueError("Situated encounter budget must be a nonnegative integer")
    if not isinstance(model["source"].get("kind"), str):
        raise ValueError("Situated source requires a kind")
    for name in ("identifiers", "assumptions", "questions"):
        values = model["source"].get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"Situated source {name} must be a list of strings")
    _json_copy(model)
    return model


class SituatedBackend:
    """One serializable world, local policy memories and encounter history."""

    def __init__(self, model_dict, seed=0, protocol=None):
        self.model = resolve_model(model_dict)
        self.seed = int(seed)
        self.protocol = _json_copy(protocol or {})
        self.provider_fingerprints = _provider_fingerprints(self.model)
        interaction = self.model["interaction"]
        provider = interaction["provider"]
        if provider == "situated" and interaction.get("game") == "ecg":
            self.interaction = ECGInteraction(interaction.get("parameters", {}))
        else:
            self.interaction = _factory(provider, interaction.get("parameters", {}))
        self.actors = tuple(self.interaction.actors)
        if not self.actors or len(set(self.actors)) != len(self.actors):
            raise ValueError("An interaction needs distinct actor identities")
        if any(not isinstance(actor, str) for actor in self.actors):
            raise ValueError("Actor identities must be strings")
        realisation = self.model["realisation"]
        adaptation = self.model["adaptation"]
        self.controllers = {}
        self.adapters = {}
        # Separate random streams preserve independence between world sampling,
        # individual policy choices, and individual adaptation.
        streams = np.random.SeedSequence(self.seed).spawn(1 + 2 * len(self.actors))
        self.world_rng = np.random.default_rng(streams[0])
        self.controller_rngs = {}
        self.adaptation_rngs = {}
        for i, actor in enumerate(self.actors):
            self.controllers[actor] = self._controller(realisation)
            self.adapters[actor] = self._adaptation(adaptation)
            self.controller_rngs[actor] = np.random.default_rng(streams[1 + 2 * i])
            self.adaptation_rngs[actor] = np.random.default_rng(streams[2 + 2 * i])
        self.controller_states = {a: None for a in self.actors}
        self.adaptation_states = {a: None for a in self.actors}
        self.observed_histories = {a: [] for a in self.actors}
        self.position = 0
        self.stages_completed = 0
        self.history: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.world = _json_copy(self.interaction.initial_world(self.world_rng))
        self.round_done = False

    @staticmethod
    def _controller(spec):
        provider, parameters = spec["provider"], spec.get("parameters", {})
        if provider == "random":
            return RandomController()
        if provider == "scripted":
            return ScriptedController(parameters)
        return _factory(provider, parameters)

    @staticmethod
    def _adaptation(spec):
        provider, parameters = spec["provider"], spec.get("parameters", {})
        if provider == "none":
            return NoAdaptation()
        if provider == "reward_trace":
            return RewardTraceAdaptation()
        return _factory(provider, parameters)

    @property
    def clock(self):
        return "encounters"

    @property
    def status(self):
        return "complete" if self.round_done else ("ready" if self.stages_completed == 0 else "running")

    def step_stage(self):
        """Execute one complete simultaneous batch; return a detached receipt."""
        if self.round_done:
            self.world = _json_copy(self.interaction.initial_world(self.world_rng))
            self.round_done = False
        pre_world = copy.deepcopy(self.world)
        stage = self.interaction.stage(copy.deepcopy(pre_world))
        if not isinstance(stage, Stage):
            raise TypeError("Interaction.stage must return Stage")
        if not stage.actors or len(set(stage.actors)) != len(stage.actors):
            raise ValueError("A stage needs distinct active actors")
        if any(a not in self.actors for a in stage.actors):
            raise ValueError("Stage contains an unknown actor")
        observations = _json_copy(self.interaction.observations(copy.deepcopy(pre_world)))
        views, actions, next_states = {}, {}, {}
        # Obtain *all* views and action spaces before invoking any policy.
        legal_spaces = {}
        for actor in stage.actors:
            views[actor] = {"actor": actor, "encounter": self.position,
                            "stage": stage.id, "observation": observations[actor],
                            "history": copy.deepcopy(self.observed_histories[actor])}
            legal_spaces[actor] = self.interaction.legal_actions(copy.deepcopy(pre_world), actor)
        for actor in stage.actors:
            action, state = self.controllers[actor].choose(
                copy.deepcopy(views[actor]), copy.deepcopy(legal_spaces[actor]),
                copy.deepcopy(self.controller_states[actor]), self.controller_rngs[actor])
            if action not in legal_spaces[actor]:
                raise ValueError(f"Illegal action for actor {actor}: {action!r}")
            actions[actor], next_states[actor] = _json_copy(action), _json_copy(state)
        result = self.interaction.transition(copy.deepcopy(pre_world), copy.deepcopy(actions), self.world_rng)
        if not isinstance(result, StageResult):
            raise TypeError("Interaction.transition must return StageResult")
        if result.terminated and result.truncated:
            raise ValueError("Termination and truncation are distinct end conditions")
        if any(a not in self.actors for a in result.feedback):
            raise ValueError("Feedback contains an unknown actor")
        self.world = _json_copy(result.world)
        self.controller_states.update(next_states)
        receipt = {"encounter": self.position, "stage": stage.id,
                   "actions": actions, "feedback": _json_copy(result.feedback),
                   "terminated": bool(result.terminated), "truncated": bool(result.truncated),
                   "info": _json_copy(result.info)}
        for actor in self.actors:
            # Non-acting actors may receive public end-of-round feedback too.
            event = {"encounter": self.position, "stage": stage.id,
                     "observation": copy.deepcopy(observations.get(actor)),
                     "acted": actor in actions, "action": copy.deepcopy(actions.get(actor)),
                     "feedback": copy.deepcopy(result.feedback.get(actor)),
                     "terminated": bool(result.terminated), "truncated": bool(result.truncated)}
            controller_state, adaptation_state = self.adapters[actor].update(
                copy.deepcopy(event), copy.deepcopy(self.controller_states[actor]),
                copy.deepcopy(self.adaptation_states[actor]), self.adaptation_rngs[actor])
            self.controller_states[actor] = _json_copy(controller_state)
            self.adaptation_states[actor] = _json_copy(adaptation_state)
            self.observed_histories[actor].append(event)
        self.history.append(copy.deepcopy(receipt))
        self.stages_completed += 1
        self.last_result = copy.deepcopy(receipt)
        self.round_done = result.terminated or result.truncated
        if self.round_done:
            self.position += 1
        return copy.deepcopy(receipt)

    def advance(self, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("Encounter advance must be a nonnegative integer")
        target = self.position + amount
        while self.position < target:
            self.step_stage()
        return self.evaluate()

    def evaluate(self, **parameters):
        if parameters:
            raise ValueError("Situated evaluation summarizes retained encounter outcomes")
        endings = [entry for entry in self.history if entry["terminated"] or entry["truncated"]]
        rewards = [float(e["info"]["reward"]) for e in endings if "reward" in e["info"]]
        return {"clock": self.clock, "position": self.position,
                "encounters": self.position, "stages": self.stages_completed,
                "terminated": sum(e["terminated"] for e in endings),
                "truncated": sum(e["truncated"] for e in endings),
                "reward_samples": len(rewards),
                "mean_reward": sum(rewards) / len(rewards) if rewards else None}

    def inspect(self):
        return {"runtime": "situated", "clock": self.clock, "position": self.position,
                "actors": list(self.actors), "components": [
                    {"id": "interaction", "kind": "interaction", "provider": self.model["interaction"]["provider"]},
                    *[{"id": actor, "owner": actor, "kind": "controller",
                       "provider": self.model["realisation"]["provider"]} for actor in self.actors],
                    *[{"id": f"adaptation:{actor}", "owner": actor, "kind": "adaptation",
                       "provider": self.model["adaptation"]["provider"]} for actor in self.actors]],
                "stage": None if self.round_done else self.interaction.stage(copy.deepcopy(self.world)).id,
                "round_done": self.round_done, "evaluation": self.evaluate()}

    def snapshot(self):
        stage = None if self.round_done else self.interaction.stage(copy.deepcopy(self.world))
        return _json_copy({"schema_version": 1, "runtime": "situated", "model": self.model,
            "seed": self.seed, "protocol": self.protocol, "position": self.position,
            "providers": self.provider_fingerprints,
            "stage": None if stage is None else {"id": stage.id, "actors": list(stage.actors)},
            "stages_completed": self.stages_completed, "world": self.world,
            "round_done": self.round_done, "history": self.history, "last_result": self.last_result,
            "controller_states": self.controller_states, "adaptation_states": self.adaptation_states,
            "observed_histories": self.observed_histories,
            "rng": {"world": self.world_rng.bit_generator.state,
                "controllers": {a: r.bit_generator.state for a, r in self.controller_rngs.items()},
                "adaptation": {a: r.bit_generator.state for a, r in self.adaptation_rngs.items()}}})


def create_backend(model_dict, seed=0, protocol=None):
    return SituatedBackend(model_dict, seed, protocol)


def restore_backend(model_dict, snapshot):
    snapshot = _json_copy(snapshot)
    if snapshot.get("schema_version") != 1 or snapshot.get("runtime") != "situated":
        raise ValueError("Unsupported situated snapshot")
    model = resolve_model(model_dict)
    if model != snapshot["model"]:
        raise ValueError("Situated checkpoint model differs")
    if snapshot.get("providers") != _provider_fingerprints(model):
        raise ValueError("Situated checkpoint external provider code differs")
    backend = SituatedBackend(model, snapshot["seed"], snapshot["protocol"])
    for name in ("position", "stages_completed", "world", "round_done", "history", "last_result",
                  "controller_states", "adaptation_states", "observed_histories"):
        setattr(backend, name, copy.deepcopy(snapshot[name]))
    backend.world_rng.bit_generator.state = snapshot["rng"]["world"]
    for actor in backend.actors:
        backend.controller_rngs[actor].bit_generator.state = snapshot["rng"]["controllers"][actor]
        backend.adaptation_rngs[actor].bit_generator.state = snapshot["rng"]["adaptation"][actor]
    if backend.snapshot()["stage"] != snapshot["stage"]:
        raise ValueError("Situated checkpoint stage differs from its retained world")
    return backend


def situated_catalogue():
    """Source-linked constructed fixture, kept separate from Barrett case IDs."""
    return [{
        "id": "ECG", "name": "Embodied colour-board coordination (discrete-time variant)",
        "family": "situated", "runtime": "situated", "profile": "ecg.demo",
        "implemented": True, "adaptive_choices": "Ordinary movement, wait and finish",
        "source_learning": "Explicit random or scripted local policies",
        "mechanism_status": "Constructed discrete-time variant",
        "required_fields": [], "source_questions": [], "learners": [],
        "validation_status": "mechanism_tests", "profile_status": "complete_demonstrations",
        "source_reproduction_status": "constructed_variant", "analysis_available": False,
        "modules": {"A": {"owner": "A", "kind": "controller"},
                    "B": {"owner": "B", "kind": "controller"}},
        "source": copy.deepcopy(ECG_MODEL["source"]),
    }]
