import json
from copy import deepcopy

import numpy as np
import pytest

from self_assembling_games.toylife.tasks import (
    resolve_task,
    static_cases,
    episode,
    phase,
    transition,
)


@pytest.mark.parametrize(
    "name",
    [
        "classification",
        "boolean",
        "context",
        "vote",
        "ordering",
        "addition",
        "sum_compare",
        "monty_hall",
    ],
)
def test_static_normalization_and_bounds(name):
    task = resolve_task(name, {})
    cases = static_cases(task)
    assert sum(c["probability"] for c in cases) == pytest.approx(1)
    for case in cases:
        assert len(case["rewards"]) == len(task["actions"])
        assert 0 <= case["probability"] <= 1
        assert all(0 <= r <= 1 for r in case["rewards"])
        assert all(v in task["fields"][k] for k, v in case["observation"].items())
    json.dumps(task)
    json.dumps(cases)


def test_exhaustive_boolean_tables():
    from itertools import product

    for table in product([0, 1], repeat=4):
        cases = static_cases(resolve_task("boolean", {"truth_table": list(table)}))
        assert [c["rewards"].index(1) for c in cases] == list(table)


@pytest.mark.parametrize("name", ["ordering", "addition", "sum_compare"])
def test_exhaustive_arithmetic(name):
    for case in static_cases(resolve_task(name, {})):
        o = case["observation"]
        if name == "addition":
            target = o["a"] + o["b"]
        else:
            a = o["a"] + o["b"] if name == "sum_compare" else o["a"]
            b = o.get("comparison", o["b"])
            target = 0 if a < b else 1 if a == b else 2
        assert case["rewards"].index(1) == target


@pytest.mark.parametrize(
    "name",
    [
        "delay",
        "boxes",
        "bandit",
        "advisers",
        "query",
        "conditioning",
        "habituation",
        "reversal",
    ],
)
def test_world_roundtrip_replay_hidden_state_and_termination(name):
    task = resolve_task(name, {})
    rng = np.random.default_rng(9)
    world = episode(task, rng)
    for _ in range(200):
        current = phase(task, world)
        assert set(current["observation"]) <= set(task["fields"])
        assert all(v in task["fields"][k] for k, v in current["observation"].items())
        assert "score" not in current["observation"]
        if current["done"]:
            break
        restored = json.loads(json.dumps(world))
        replica_rng = np.random.default_rng()
        replica_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        action = current["legal_actions"][-1] if current["act"] else None
        result = transition(task, world, action, rng)
        assert transition(task, restored, action, replica_rng) == result
        assert phase(task, world) == current
        world, reward, _ = result
        assert reward is None or 0 <= reward <= 1
    else:
        pytest.fail("episode did not terminate")


def test_delay_has_identity_then_absence_then_zero_duration_read():
    task = resolve_task("delay", {})
    rng = np.random.default_rng(1)
    w = episode(task, rng)
    assert phase(task, w)["observation"]["cue"] in (0, 1)
    w, _, _ = transition(task, w, None, rng)
    assert "cue" not in phase(task, w)["observation"]
    w, _, _ = transition(task, w, None, rng)
    assert phase(task, w)["duration"] == 0 and phase(task, w)["act"]


def test_query_attributes_and_terminal_material():
    task = resolve_task("query", {})
    rng = np.random.default_rng(3)
    w = episode(task, rng)
    for feature in [0, 1]:
        w, reward, _ = transition(task, w, feature, rng)
        assert reward is None
        assert phase(task, w)["observation"]["ab"[feature]] == w["features"][feature]
    assert phase(task, w)["legal_actions"] == [2, 3, 4, 5]
    w, reward, _ = transition(task, w, 2 + 2 * w["features"][0] + w["features"][1], rng)
    assert reward == 1
    assert phase(task, w)["observation"]["reward"] == 1
    assert not phase(task, w)["act"]


@pytest.mark.parametrize(
    "name,params",
    [
        ("classification", {"size": 0}),
        ("boxes", {"mode": "bad"}),
        ("vote", {"error": float("nan")}),
        ("boolean", {"truth_table": [0]}),
        ("bandit", {"probabilities": [1]}),
        ("delay", {"extra": 1}),
        ("reversal", {"trials": True}),
    ],
)
def test_invalid_parameters(name, params):
    with pytest.raises(ValueError):
        resolve_task(name, params)


def test_query_exposes_only_current_answer_and_rejects_repeat():
    task = resolve_task("query", {})
    rng = np.random.default_rng(5)
    w = episode(task, rng)
    assert set(phase(task, w)["observation"]) == {"reward"}
    w, _, _ = transition(task, w, 0, rng)
    assert "a" in phase(task, w)["observation"]
    original = deepcopy(w)
    with pytest.raises(ValueError):
        transition(task, w, 0, rng)
    assert w == original
    w, _, _ = transition(task, w, 1, rng)
    assert "b" in phase(task, w)["observation"]
    assert "a" not in phase(task, w)["observation"]


def test_conditioning_matched_doses_and_absent_cue():
    totals = []
    for mode in ["paired", "unpaired"]:
        task = resolve_task("conditioning", {"mode": mode, "duration": 3})
        rng = np.random.default_rng(1)
        w = episode(task, rng)
        cue = food = elapsed = 0
        while not phase(task, w)["act"]:
            current = phase(task, w)
            o = current["observation"]
            d = current["duration"]
            cue += o.get("cue", 0) * d
            food += o["food"] * d
            elapsed += d
            if not o.get("cue"):
                assert "cue" not in o
            w, _, _ = transition(task, w, None, rng)
        totals.append((cue, food, elapsed))
    assert totals == [(12, 12, 32)] * 2


def test_ordering_adjacent_and_held_out_partitions():
    from self_assembling_games.toylife.tasks import held_out_cases

    task = resolve_task("ordering", {"train_pairs": "adjacent"})
    train = static_cases(task)
    held = held_out_cases(task)
    assert len(train) == 12 and len(held) == 30
    assert all(abs(c["observation"]["a"] - c["observation"]["b"]) == 1 for c in train)
    assert all(abs(c["observation"]["a"] - c["observation"]["b"]) > 1 for c in held)
    assert sum(c["probability"] for c in train) == pytest.approx(1)
    assert sum(c["probability"] for c in held) == pytest.approx(1)


def test_habituation_constant_suppression_baseline_and_diagnostics():
    task = resolve_task("habituation", {})
    rng = np.random.default_rng(4)
    w = episode(task, rng)
    rewards = []
    diagnostics = []
    while not phase(task, w)["done"]:
        current = phase(task, w)
        w, reward, info = transition(task, w, 0 if current["act"] else None, rng)
        if reward is not None:
            rewards.append(reward)
            diagnostics.append(info)
    assert rewards == [1.0] * 5
    assert w["responses"] == [0] * 5
    assert diagnostics[-1]["recovery_response"] == 0
    assert "constant-zero" in diagnostics[-1]["objective"]


def test_task_interaction_labels_feedback_and_detached_metadata():
    from self_assembling_games.toylife.tasks import TaskInteraction

    task = resolve_task("boxes", {"trials": 1})
    interaction = TaskInteraction(task)
    task["actions"].append("external")
    assert interaction.actions == ["0", "1"]
    rng = np.random.default_rng(2)
    w = interaction.initial_world(rng)
    assert interaction.legal_actions(w, "cell") == ["0", "1"]
    result = interaction.transition(w, {"cell": str(w["target"])}, rng)
    assert result.feedback == {"cell": {"score": 1.0}}
    assert interaction.observations(result.world)["cell"]["reward"] == 1.0
    assert interaction.legal_actions(result.world, "cell") == []
    final = interaction.transition(result.world, {}, rng)
    assert final.terminated
    with pytest.raises(ValueError):
        interaction.transition(w, {}, rng)
    with pytest.raises(ValueError):
        interaction.legal_actions(w, "other")


def test_static_interaction_metadata_and_stage_duration():
    from self_assembling_games.toylife.tasks import TaskInteraction

    static = TaskInteraction(resolve_task("boolean", {}))
    assert static.actions == ["0", "1"]
    interaction = TaskInteraction(resolve_task("delay", {"cue_updates": 3}))
    rng = np.random.default_rng(1)
    w = interaction.initial_world(rng)
    assert interaction.stage(w).duration == 3
    assert interaction.stage(w).actors == ()
    for _ in range(2):
        w = interaction.transition(w, {}, rng).world
    assert interaction.stage(w).duration == 0
    assert interaction.stage(w).actors == ("cell",)


@pytest.mark.parametrize(
    "name",
    [
        "classification",
        "boolean",
        "context",
        "vote",
        "ordering",
        "addition",
        "sum_compare",
        "monty_hall",
        "delay",
        "boxes",
        "bandit",
        "advisers",
        "query",
        "conditioning",
        "habituation",
        "reversal",
    ],
)
def test_all_phase_observations_encode(name):
    from self_assembling_games.toylife.apparatus import resolve_apparatus, encode

    task = resolve_task(name, {})
    apparatus = resolve_apparatus(task["fields"], task["actions"], {})
    if task["static"]:
        for case in static_cases(task):
            encode(apparatus, case["observation"])
        return
    rng = np.random.default_rng(12)
    for _ in range(8):
        w = episode(task, rng)
        while True:
            current = phase(task, w)
            encoded = encode(apparatus, current["observation"])
            assert all(0 <= v <= apparatus["max_dose"] for v in encoded.values())
            if current["done"]:
                break
            action = (
                int(rng.choice(current["legal_actions"])) if current["act"] else None
            )
            w, _, _ = transition(task, w, action, rng)


def test_random_boxes_fixed_special_and_probabilities():
    class ControlledRng:
        def __init__(self, value):
            self.value = value

        def integers(self, n):
            return 0

        def random(self):
            return self.value

    task = resolve_task("boxes", {"mode": "random", "trials": 2})
    for action, draw, expected in [
        (0, 0.6, 1.0),
        (0, 0.7, 0.0),
        (1, 0.3, 1.0),
        (1, 0.4, 0.0),
    ]:
        rng = ControlledRng(draw)
        w = episode(task, rng)
        w, reward, _ = transition(task, w, action, rng)
        assert reward == expected
        w, _, _ = transition(task, w, None, rng)
        assert w["target"] == 0


@pytest.mark.parametrize("order", ["cue_food", "food_cue", "counterbalanced"])
def test_conditioning_common_washout_and_order_controls(order):
    worlds = []
    for mode in ["paired", "unpaired"]:
        task = resolve_task(
            "conditioning", {"mode": mode, "order": order, "washout": 9}
        )
        worlds.append(episode(task, np.random.default_rng(8)))
    for world in worlds:
        assert world["schedule"][-2] == {
            "cue": 0,
            "food": 0,
            "duration": 9,
            "act": False,
        }
        assert len(world["schedule"]) == 10
    unpaired = worlds[1]["schedule"][:-2]
    first_cues = [unpaired[i]["cue"] for i in range(0, 8, 2)]
    if order == "counterbalanced":
        assert sum(first_cues) == 2
        assert all(a != b for a, b in zip(first_cues, first_cues[1:]))
    else:
        assert first_cues == [int(order == "cue_food")] * 4
    assert all(step["cue"] + step["food"] == 1 for step in unpaired)


@pytest.mark.parametrize("control", ["none", "intact", "yoked"])
def test_habituation_feedback_controls_keep_timing_and_diagnostics(control):
    parameters = {"feedback": control, "novel_probe": True}
    if control == "yoked":
        parameters["yoked_rewards"] = [1, 0, 1, 0, 1, 0]
    task = resolve_task("habituation", parameters)
    rng = np.random.default_rng(6)
    world = episode(task, rng)
    feedback, cues, diagnostics = [], [], []
    while not phase(task, world)["done"]:
        current = phase(task, world)
        if world["stage"] == "feedback":
            assert current["duration"] == 1
            feedback.append(current["observation"]["reward"])
        if current["act"]:
            cues.append(current["observation"]["cue"])
        world, score, info = transition(task, world, 1 if current["act"] else None, rng)
        if score is not None:
            diagnostics.append(info)
    assert cues == [1] * 5 + [2]
    assert (
        feedback
        == (
            {"none": ["none"] * 6, "intact": [0.0] * 6, "yoked": [1, 0, 1, 0, 1, 0]}[
                control
            ]
        )
    )
    assert diagnostics[0]["initial_response"] == 1
    assert diagnostics[-2]["recovery_response"] == 1
    assert diagnostics[-1]["novel_response"] == 1


@pytest.mark.parametrize("mode", ["informative", "blank", "inverted"])
def test_delay_age_matched_cue_controls(mode):
    task = resolve_task("delay", {"cue_mode": mode})
    rng = np.random.default_rng(3)
    world = episode(task, rng)
    current = phase(task, world)
    if mode == "blank":
        assert "cue" not in current["observation"]
    else:
        assert current["observation"]["cue"] == world["target"] ^ int(
            mode == "inverted"
        )
    assert current["duration"] == 8
    world, _, _ = transition(task, world, None, rng)
    assert phase(task, world)["duration"] == 8
    assert "cue" not in phase(task, world)["observation"]
    world, _, _ = transition(task, world, None, rng)
    _, score, _ = transition(task, world, world["target"], rng)
    assert score == 1


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("conditioning", {"order": "bad"}),
        ("conditioning", {"washout": -1}),
        ("habituation", {"feedback": "bad"}),
        ("habituation", {"feedback": "yoked", "yoked_rewards": [1]}),
        ("habituation", {"yoked_rewards": [1] * 5}),
        ("habituation", {"novel_probe": 1}),
        ("delay", {"cue_mode": "bad"}),
    ],
)
def test_invalid_assay_controls(name, parameters):
    with pytest.raises(ValueError):
        resolve_task(name, parameters)


def test_yoked_ligands_are_independent_of_recipient_responses():
    task = resolve_task(
        "habituation", {"feedback": "yoked", "yoked_rewards": [1, 0, 0, 1, 0]}
    )
    traces, scores = [], []
    for action in [0, 1]:
        rng = np.random.default_rng(9)
        world = episode(task, rng)
        trace, score_trace = [], []
        while not phase(task, world)["done"]:
            current = phase(task, world)
            trace.append((current["observation"], current["duration"]))
            world, score, _ = transition(
                task, world, action if current["act"] else None, rng
            )
            if score is not None:
                score_trace.append(score)
        traces.append(trace)
        scores.append(score_trace)
    assert traces[0] == traces[1]
    assert scores == [[1.0] * 5, [0.0] * 5]
