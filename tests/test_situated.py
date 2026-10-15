"""Behavioral checks for staged private observations and checkpoint continuation."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from self_assembling_games.situated import (
    ECGInteraction, Stage, StageResult, create_backend, resolve_model, restore_backend,
)


def scripted(scripts, **parameters):
    return resolve_model("ecg.demo", {
        "realisation": {"provider": "scripted", "parameters": {"scripts": scripts}},
        "interaction": {"parameters": parameters},
    })


def test_no_communicative_roles_or_designated_signal_actions():
    backend = create_backend("ecg.demo")
    assert backend.actors == ("A", "B")
    assert backend.interaction.actions == ("up", "down", "left", "right", "wait", "finish")
    assert all("role" not in component for component in backend.inspect()["components"])


def test_private_view_exposes_own_board_positions_and_finished_only():
    backend = create_backend("ecg.demo", seed=42)
    world = copy.deepcopy(backend.world)
    views = backend.interaction.observations(world)
    assert views["A"]["board"] == world["boards"]["A"]
    assert set(views["A"]) == {"board", "size", "positions", "finished", "tick"}
    assert views["A"]["positions"] == world["positions"]
    views["A"]["board"][0] = "modified"
    views["A"]["positions"]["B"] = -1
    assert backend.world == world


def test_boards_have_a_common_colour_and_positions_are_valid():
    backend = create_backend("ecg.demo", seed=9)
    for _ in range(100):
        world = backend.interaction.initial_world(backend.world_rng)
        assert set(world["boards"]["A"]) & set(world["boards"]["B"])
        assert all(0 <= position < 4 for position in world["positions"].values())


def test_boundaries_wait_and_finish_are_ordinary_actions():
    interaction = ECGInteraction({})
    backend = create_backend("ecg.demo")
    world = backend.world
    world["positions"] = {"A": 0, "B": 3}
    result = interaction.transition(world, {"A": "up", "B": "right"}, backend.world_rng)
    assert result.world["positions"] == {"A": 0, "B": 3}
    result = interaction.transition(result.world, {"A": "wait", "B": "left"}, backend.world_rng)
    assert result.world["positions"] == {"A": 0, "B": 2}
    result = interaction.transition(result.world, {"A": "down", "B": "up"}, backend.world_rng)
    assert result.world["positions"] == {"A": 2, "B": 0}
    result = interaction.transition(result.world, {"A": "right", "B": "finish"}, backend.world_rng)
    assert result.world["positions"] == {"A": 3, "B": 0}
    assert interaction.stage(result.world).actors == ("A",)
    assert not result.terminated
    result = interaction.transition(result.world, {"A": "finish"}, backend.world_rng)
    assert result.terminated and not result.truncated
    assert result.feedback["A"]["boards"] == result.world["boards"]
    assert result.feedback["A"]["reward"] == result.feedback["B"]["reward"]


def test_reward_matches_final_cells():
    backend = create_backend(scripted({"A": ["finish"], "B": ["finish"]}))
    backend.world["positions"] = {"A": 1, "B": 3}
    backend.world["boards"] = {"A": ["red", "blue", "red", "red"],
                                "B": ["red", "red", "red", "blue"]}
    receipt = backend.step_stage()
    assert receipt["feedback"]["A"]["reward"] == 1
    assert receipt["feedback"]["B"]["reward"] == 1
    assert backend.position == 1


def test_tick_cap_is_truncation_with_no_terminal_reward_or_reveal():
    backend = create_backend(scripted({"A": ["wait"], "B": ["wait"]}, tick_cap=1))
    receipt = backend.step_stage()
    assert receipt["truncated"] and not receipt["terminated"]
    assert receipt["feedback"] == {"A": {}, "B": {}}
    assert backend.evaluate()["truncated"] == 1
    assert backend.evaluate()["reward_samples"] == 0
    assert backend.position == 1


def test_finish_at_tick_cap_terminates():
    backend = create_backend(scripted({"A": ["finish"], "B": ["finish"]}, tick_cap=1))
    receipt = backend.step_stage()
    assert receipt["terminated"] and not receipt["truncated"]


class ViewProbeController:
    def choose(self, view, legal_actions, state, rng):
        original = copy.deepcopy(view)
        # Mutating one controller's input cannot modify another actor or world.
        if view["actor"] == "A":
            view["observation"]["positions"]["A"] = -999
            view["observation"]["board"][0] = "poison"
        state = {"view": original}
        return ("right" if view["actor"] == "A" else "wait"), state


def probe_controller(parameters):
    return ViewProbeController()


def test_simultaneous_views_and_policy_inputs_are_detached():
    model = resolve_model("ecg.demo", {
        "realisation": {"provider": "test_situated:probe_controller"}})
    backend = create_backend(model)
    backend.world["positions"] = {"A": 0, "B": 3}
    board = copy.deepcopy(backend.world["boards"]["A"])
    backend.step_stage()
    seen_a = backend.controller_states["A"]["view"]["observation"]
    seen_b = backend.controller_states["B"]["view"]["observation"]
    assert seen_a["positions"] == seen_b["positions"] == {"A": 0, "B": 3}
    assert backend.world["positions"] == {"A": 1, "B": 3}
    assert backend.world["boards"]["A"] == board
    assert seen_b["board"] == backend.world["boards"]["B"]


def test_full_json_restore_mid_round_and_across_rounds():
    model = scripted({"A": ["right", "finish"], "B": ["wait", "finish"]}, tick_cap=5)
    model["adaptation"] = {"provider": "reward_trace", "parameters": {}}
    uninterrupted = create_backend(model, seed=19)
    uninterrupted.step_stage()
    assert uninterrupted.position == 0
    snapshot = json.loads(json.dumps(uninterrupted.snapshot()))
    restored = restore_backend(model, snapshot)
    uninterrupted.advance(4)
    restored.advance(4)
    assert restored.snapshot() == uninterrupted.snapshot()
    assert restored.position == 4
    assert restored.controller_states["A"]["index"] > 2
    assert restored.adaptation_states["A"]["updates"] == restored.stages_completed
    assert len(restored.observed_histories["A"]) == restored.stages_completed


def test_random_rng_state_restores():
    model = resolve_model("ecg.demo")
    first = create_backend(model, seed=21)
    first.step_stage()
    second = restore_backend(model, first.snapshot())
    first.advance(7)
    second.advance(7)
    assert first.snapshot() == second.snapshot()


def test_snapshot_and_receipts_are_detached():
    backend = create_backend("ecg.demo")
    receipt = backend.step_stage()
    snapshot = backend.snapshot()
    receipt["actions"]["A"] = "poison"
    snapshot["world"]["boards"]["A"][0] = "poison"
    assert backend.history[-1]["actions"]["A"] != "poison"
    assert backend.world["boards"]["A"][0] != "poison"


class HugeInteraction:
    actors = ("creature",)

    def initial_world(self, rng):
        return {"private": "secret", "tick": 0}

    def stage(self, world):
        return Stage("choice", self.actors)

    def observations(self, world):
        return {"creature": {"tick": world["tick"]}}

    def legal_actions(self, world, actor):
        return range(10**12)

    def transition(self, world, actions, rng):
        world["tick"] += 1
        return StageResult(world, {"creature": {"reward": 1}}, terminated=True)


def huge_interaction(parameters):
    return HugeInteraction()


def test_sampled_large_action_space_requires_no_enumeration():
    model = resolve_model("ecg.demo", {"interaction": {
        "provider": "test_situated:huge_interaction", "game": "large", "parameters": {}}})
    backend = create_backend(model)
    backend.advance(2)
    assert backend.position == 2
    assert all(0 <= event["actions"]["creature"] < 10**12 for event in backend.history)
    assert "secret" not in json.dumps(backend.observed_histories)


@pytest.mark.parametrize("component,factory", [
    ("interaction", "huge_interaction"), ("realisation", "probe_controller"),
])
def test_fresh_process_custom_factory_restore(tmp_path, component, factory):
    model = resolve_model("ecg.demo", {component: {"provider": f"test_situated:{factory}"},
                                      "interaction": {"parameters": {"tick_cap": 2}}}
                          if component == "realisation" else {
                              "interaction": {"provider": f"test_situated:{factory}",
                                              "game": "large", "parameters": {}}})
    backend = create_backend(model, seed=7)
    backend.advance(1)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(backend.snapshot()))
    code = (
        "import json,sys; from self_assembling_games.situated import restore_backend; "
        "s=json.load(open(sys.argv[1])); b=restore_backend(s['model'],s); "
        "b.advance(2); print(json.dumps(b.snapshot(),sort_keys=True))"
    )
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "tests")])
    process = subprocess.run([sys.executable, "-c", code, str(snapshot_path)], env=env,
                             check=True, text=True, capture_output=True)
    backend.advance(2)
    assert json.loads(process.stdout) == backend.snapshot()


def test_custom_controller_fresh_restore(tmp_path):
    model = resolve_model("ecg.demo", {"realisation": {"provider": "test_situated:probe_controller"},
                                       "interaction": {"parameters": {"tick_cap": 2}}})
    backend = create_backend(model, seed=3)
    backend.step_stage()
    restored = restore_backend(model, json.loads(json.dumps(backend.snapshot())))
    backend.advance(2)
    restored.advance(2)
    assert backend.snapshot() == restored.snapshot()


@pytest.mark.parametrize("amount", [-1, 0.5, True])
def test_bad_encounter_advance_is_rejected(amount):
    with pytest.raises(ValueError):
        create_backend("ecg.demo").advance(amount)


def test_restore_rejects_different_model():
    backend = create_backend("ecg.demo")
    changed = resolve_model("ecg.demo", {"interaction": {"parameters": {"tick_cap": 2}}})
    with pytest.raises(ValueError, match="model differs"):
        restore_backend(changed, backend.snapshot())


def test_public_model_system_midstage_restore_and_nonmutating_evaluate(tmp_path):
    from self_assembling_games import Model, System, model

    declaration = model("ecg", realisation={"provider": "scripted", "parameters": {
        "scripts": {"A": ["wait", "finish"], "B": ["right", "finish"]}}})
    assert isinstance(declaration, Model)
    system = declaration.instantiate(seed=17)
    assert system.clock == "encounters"
    assert system.status == "ready"
    system.step_stage()
    assert system.position == 0
    assert system.status == "running"
    original = system.snapshot()
    assert system.evaluate()["stages"] == 1
    assert system.snapshot() == original
    checkpoint = tmp_path / "system.json"
    system.checkpoint(checkpoint)
    restored = System.restore(checkpoint)
    restored.advance(3)
    system.advance(3)
    assert restored.snapshot() == system.snapshot()
    assert system.position == 3
    assert system.status == "complete"
    inspected = system.inspect()
    inspected["actors"].append("poison")
    assert system.inspect()["actors"] == ["A", "B"]


@pytest.mark.parametrize("component,factory", [
    ("interaction", "huge_interaction"),
    ("realisation", "probe_controller"),
    ("adaptation", "no_adaptation"),
])
def test_restore_rejects_changed_external_component_code(tmp_path, monkeypatch, component, factory):
    module_name = f"custom_situated_{component}"
    module = tmp_path / f"{module_name}.py"
    source = "self_assembling_games.situated" if component == "adaptation" else "test_situated"
    module.write_text(f"from {source} import {factory}\nfactory = {factory}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    model = resolve_model("ecg.demo", {component: {"provider": f"{module_name}:factory"}})
    backend = create_backend(model)
    snapshot = backend.snapshot()
    module.write_text(module.read_text() + "\n# changed provider source\n")
    with pytest.raises(ValueError, match="external provider code differs"):
        restore_backend(model, snapshot)


def test_outcome_denominators_include_truncations_and_duration():
    backend = create_backend(scripted({"A": ["finish"], "B": ["finish"]}, tick_cap=2))
    backend.world["boards"] = {a: ["red"] * 4 for a in backend.actors}
    backend.advance(2)
    metrics = backend.evaluate()
    assert metrics["mean_reward"] == 1
    assert metrics["reward_samples"] == 1
    assert metrics["mean_reward_per_ended_encounter_zero_filled"] == 0.5
    assert metrics["ecg_success_rate"] == 0.5
    assert metrics["completion_rate"] == 0.5
    assert metrics["encounter_durations_stages"] == [1, 2]
    assert metrics["mean_encounter_duration_stages"] == 1.5
    backend.step_stage()
    assert backend.evaluate()["open_encounter_duration_stages"] == 1


def test_generic_reward_is_not_reported_as_success():
    backend = create_backend(resolve_model("ecg.demo", {"interaction": {
        "provider": "test_situated:huge_interaction", "parameters": {}}}))
    backend.advance(1)
    assert "ecg_success_rate" not in backend.evaluate()
    assert backend.evaluate()["mean_reward"] is None


@pytest.mark.parametrize("limit", [None, 0, 3])
def test_history_window_preserves_archive_and_checkpoint_continuation(limit):
    model = resolve_model("ecg.demo", {
        "realisation": {"provider": "test_situated:probe_controller"},
        "interaction": {"parameters": {"tick_cap": 2}},
        "execution": {"parameters": {"controller_history_limit": limit}}})
    backend = create_backend(model, seed=3)
    # The probe retains its latest view, so inspect precisely what a policy received.
    for i in range(100):
        backend.step_stage()
        expected = backend.observed_histories["A"][:i]
        if limit is not None:
            expected = expected[-limit:] if limit else []
        assert backend.controller_states["A"]["view"]["history"] == expected
    assert len(backend.observed_histories["A"]) == 100
    restored = restore_backend(model, json.loads(json.dumps(backend.snapshot())))
    backend.advance(2)
    restored.advance(2)
    assert backend.snapshot() == restored.snapshot()


def test_default_and_explicit_unlimited_controller_views_are_equivalent():
    overrides = {"realisation": {"provider": "test_situated:probe_controller"},
                 "interaction": {"parameters": {"tick_cap": 2}}}
    first = create_backend(resolve_model("ecg.demo", overrides), seed=9)
    overrides["execution"] = {"parameters": {"controller_history_limit": None}}
    second = create_backend(resolve_model("ecg.demo", overrides), seed=9)
    for _ in range(8):
        assert first.step_stage() == second.step_stage()
        assert first.controller_states == second.controller_states
        assert first.observed_histories == second.observed_histories


@pytest.mark.parametrize("limit", [-1, True, 1.5, "3"])
def test_invalid_history_limit(limit):
    with pytest.raises(ValueError, match="controller_history_limit"):
        resolve_model("ecg.demo", {
            "execution": {"parameters": {"controller_history_limit": limit}}})


def test_empty_and_all_truncated_evaluation():
    backend = create_backend(scripted({}, tick_cap=1))
    initial = backend.evaluate()
    assert initial["completion_rate"] is None
    assert initial["ecg_success_rate"] is None
    assert initial["mean_reward_per_ended_encounter_zero_filled"] is None
    assert initial["encounter_durations_stages"] == []
    backend.advance(3)
    metrics = backend.evaluate()
    assert metrics["mean_reward"] is None
    assert metrics["completion_rate"] == metrics["ecg_success_rate"] == 0
    assert metrics["mean_reward_per_ended_encounter_zero_filled"] == 0
    assert metrics["encounter_durations_stages"] == [1, 1, 1]
