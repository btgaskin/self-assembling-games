"""Finite chemical-controller tasks: deliberate variants of the source games.

Static cases enumerate the complete uniform input distribution. Episodic tasks
keep hidden outcomes in JSON worlds; observed reward is a separate one-update
feedback phase. No task updates a learner. Query identifies four materials from two fixed binary attributes;
conditioning and habituation are explicit cue/probe schedules for response tests.
The reward value "none" is an explicit baseline ligand outside feedback. Food
uses presence encoding (0 means no ligand); cue absence omits the cue field.
Habituation reward measures suppression: a constant-zero controller maximizes it.
Response traces and the post-rest recovery response support behavioral comparison;
fitness alone does not establish decrement followed by recovery.
"""

from copy import deepcopy
from itertools import product
import math

_DEFAULTS = {
    "classification": {"size": 2},
    "boolean": {"truth_table": [1, 1, 1, 0]},
    "context": {},
    "vote": {"error": 0.1},
    "ordering": {"size": 7, "train_pairs": "all"},
    "addition": {"maximum": 5},
    "sum_compare": {"maximum": 5, "comparison_max": 10},
    "monty_hall": {},
    "delay": {"cue_updates": 8, "delay": 8, "cue_mode": "informative"},
    "boxes": {"mode": "fixed", "trials": 10},
    "bandit": {"probabilities": [0.2, 0.5, 0.8], "trials": 10},
    "advisers": {"probabilities": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "trials": 10},
    "query": {"max_queries": 2},
    "conditioning": {
        "mode": "paired",
        "repetitions": 4,
        "duration": 1,
        "washout": 8,
        "order": "counterbalanced",
    },
    "habituation": {
        "repetitions": 4,
        "rest": 8,
        "feedback": "none",
        "yoked_rewards": [],
        "novel_probe": False,
    },
    "reversal": {"trials": 10},
}
_STATIC = set(list(_DEFAULTS)[:8])


def resolve_task(name: str, parameters: dict) -> dict:
    """Resolve validated parameters and the complete scalar observation alphabet."""
    if name not in _DEFAULTS:
        raise ValueError(f"unknown task {name!r}")
    if not isinstance(parameters, dict) or set(parameters) - set(_DEFAULTS[name]):
        raise ValueError("unknown task parameters")
    p = deepcopy(_DEFAULTS[name])
    p.update(deepcopy(parameters))
    for k, v in p.items():
        if k in {
            "size",
            "maximum",
            "comparison_max",
            "trials",
            "cue_updates",
            "delay",
            "max_queries",
            "repetitions",
            "duration",
            "rest",
            "washout",
        }:
            minimum = (
                0
                if k in {"maximum", "comparison_max", "delay", "rest", "washout"}
                else 1
            )
            if type(v) is not int or v < minimum:
                raise ValueError(f"{k} must be an integer >= {minimum}")
        elif k == "cue_mode":
            if v not in {"informative", "blank", "inverted"}:
                raise ValueError("invalid delay cue mode")
        elif k == "order":
            if v not in {"counterbalanced", "cue_food", "food_cue"}:
                raise ValueError("invalid exposure order")
        elif k == "feedback":
            if v not in {"none", "intact", "yoked"}:
                raise ValueError("invalid feedback control")
        elif k == "novel_probe":
            if type(v) is not bool:
                raise ValueError("novel_probe must be boolean")
        elif k == "yoked_rewards":
            if not isinstance(v, list) or any(
                type(x) is not int or x not in (0, 1) for x in v
            ):
                raise ValueError("yoked_rewards must contain binary integers")
        elif k == "train_pairs":
            if v not in {"all", "adjacent"}:
                raise ValueError("train_pairs must be all or adjacent")
            if v == "adjacent" and p["size"] < 2:
                raise ValueError("adjacent ordering needs size >= 2")
        elif k == "error":
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 <= v <= 1
            ):
                raise ValueError("error must lie in [0,1]")
        elif k == "truth_table":
            if (
                not isinstance(v, list)
                or len(v) != 4
                or any(type(x) is not int or x not in (0, 1) for x in v)
            ):
                raise ValueError("truth_table must contain four binary integers")
        elif k == "probabilities":
            count = 3 if name == "bandit" else 7
            if (
                not isinstance(v, list)
                or len(v) != count
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    or not 0 <= x <= 1
                    for x in v
                )
            ):
                raise ValueError(f"probabilities must contain {count} values in [0,1]")
        elif k == "mode" and v not in (
            {"fixed", "random"} if name == "boxes" else {"paired", "unpaired"}
        ):
            raise ValueError("invalid mode")
    if name == "habituation":
        expected = p["repetitions"] + 1 + int(p["novel_probe"])
        if p["feedback"] == "yoked" and len(p["yoked_rewards"]) != expected:
            raise ValueError("yoked_rewards needs one value per scored presentation")
        if p["feedback"] != "yoked" and p["yoked_rewards"]:
            raise ValueError("yoked_rewards requires feedback=yoked")
    actions = ["0", "1"]
    fields = {}
    if name == "classification":
        actions = [str(i) for i in range(p["size"])]
        fields = {"stimulus": list(range(p["size"]))}
    elif name in {"boolean", "addition", "ordering", "sum_compare"}:
        maximum = p.get("maximum", p.get("size", 2) - 1)
        fields = {"a": list(range(maximum + 1)), "b": list(range(maximum + 1))}
        if name == "addition":
            actions = [str(i) for i in range(2 * maximum + 1)]
        if name in {"ordering", "sum_compare"}:
            actions = ["less", "equal", "greater"]
        if name == "sum_compare":
            fields["comparison"] = list(range(p["comparison_max"] + 1))
    if name == "boolean":
        fields = {"x": [0, 1], "y": [0, 1]}
    elif name == "context":
        fields = {"a": [0, 1], "b": [0, 1], "context": [0, 1]}
    elif name == "vote":
        fields = {f"vote_{i}": [0, 1] for i in range(3)}
    elif name == "monty_hall":
        actions = ["stay", "switch"]
    elif name == "delay":
        fields = {"cue": [0, 1]}
    elif name in {"bandit", "advisers"}:
        actions = [str(i) for i in range(len(p["probabilities"]))]
    elif name == "query":
        actions = [
            "query_a",
            "query_b",
            "material_0",
            "material_1",
            "material_2",
            "material_3",
        ]
        fields = {"a": ["unknown", 0, 1], "b": ["unknown", 0, 1]}
    elif name in {"conditioning", "habituation"}:
        fields = {"cue": [0, 1, 2] if p.get("novel_probe") else [0, 1], "food": [0, 1]}
    elif name == "reversal":
        fields = {"cue": [0, 1]}
    if name not in _STATIC:
        fields["reward"] = ["none", 0.0, 1.0]
    return {
        "name": name,
        "parameters": p,
        "actions": actions,
        "fields": fields,
        "static": name in _STATIC,
    }


def static_cases(task: dict) -> list:
    if not task["static"]:
        raise ValueError("static_cases requires a static task")
    name, p = task["name"], task["parameters"]
    observations = [
        dict(zip(task["fields"], values))
        for values in product(*task["fields"].values())
    ]
    if name == "ordering" and p["train_pairs"] == "adjacent":
        observations = [o for o in observations if abs(o["a"] - o["b"]) == 1]
    cases = []
    for o in observations:
        probability = 1 / len(observations)
        if name == "monty_hall":
            # Uniform prize and initial choice; host always opens an unchosen goat.
            rewards = [1 / 3, 2 / 3]
        elif name == "vote":
            e = p["error"]
            ones = sum(o.values())
            weights = [
                (1 - e) ** (3 - ones) * e**ones,
                (1 - e) ** ones * e ** (3 - ones),
            ]
            probability = sum(weights) / 2
            rewards = [w / sum(weights) if sum(weights) else 0.5 for w in weights]
        else:
            if name == "classification":
                target = o["stimulus"]
            elif name == "boolean":
                target = p["truth_table"][2 * o["x"] + o["y"]]
            elif name == "context":
                target = o["b"] if o["context"] else o["a"]
            elif name == "addition":
                target = o["a"] + o["b"]
            else:
                left = o["a"] + o["b"] if name == "sum_compare" else o["a"]
                right = o["comparison"] if name == "sum_compare" else o["b"]
                target = 0 if left < right else 1 if left == right else 2
            rewards = [float(i == target) for i in range(len(task["actions"]))]
        cases.append({"observation": o, "rewards": rewards, "probability": probability})
    return cases


def held_out_cases(task: dict) -> list:
    """Uniform nonadjacent unequal ordering pairs, disjoint from adjacent training."""
    if task["name"] != "ordering":
        raise ValueError("held_out_cases requires ordering")
    full = deepcopy(task)
    full["parameters"]["train_pairs"] = "all"
    cases = [
        c
        for c in static_cases(full)
        if abs(c["observation"]["a"] - c["observation"]["b"]) > 1
    ]
    for case in cases:
        case["probability"] = 1 / len(cases)
    return cases


def episode(task: dict, rng) -> dict:
    if task["static"]:
        raise ValueError("episode requires an episodic task")
    name, p = task["name"], task["parameters"]
    w = {"stage": "choice", "trial": 0, "done": False, "score": 0.0, "feedback": None}
    if name in {"boxes", "reversal", "delay"}:
        w["target"] = int(rng.integers(2))
    if name == "reversal":
        w["cue"] = int(rng.integers(2))
    if name in {"bandit", "advisers"}:
        w["probabilities"] = list(p["probabilities"])
        w["probabilities"] = rng.permutation(w["probabilities"]).tolist()
    if name == "query":
        w.update(
            features=[int(rng.integers(2)), int(rng.integers(2))],
            known=["unknown", "unknown"],
            queries=0,
            latest=None,
        )
    if name == "delay":
        w["stage"] = "cue"
    if name == "habituation":
        w["responses"] = []
    if name in {"conditioning", "habituation"}:
        schedule = []
        start_reverse = (
            bool(rng.integers(2)) if p.get("order") == "counterbalanced" else False
        )
        for repetition in range(p["repetitions"]):
            cycle_start = len(schedule)
            if name == "habituation":
                schedule.append({"cue": 1, "food": 0, "duration": 1, "act": True})
            elif p["mode"] == "paired":
                schedule.extend(
                    [
                        {"cue": 1, "food": 1, "duration": p["duration"], "act": False},
                        {"cue": 0, "food": 0, "duration": p["duration"], "act": False},
                    ]
                )
            else:
                schedule.extend(
                    [
                        {"cue": 1, "food": 0, "duration": p["duration"], "act": False},
                        {"cue": 0, "food": 1, "duration": p["duration"], "act": False},
                    ]
                )
            if name == "conditioning":
                reverse = p["order"] == "food_cue" or (
                    p["order"] == "counterbalanced"
                    and (repetition + start_reverse) % 2 == 1
                )
                if reverse:
                    schedule[cycle_start:] = reversed(schedule[cycle_start:])
        if name == "conditioning":
            schedule.append(
                {"cue": 0, "food": 0, "duration": p["washout"], "act": False}
            )
        if name == "habituation":
            schedule.append({"cue": 0, "food": 0, "duration": p["rest"], "act": False})
        schedule.append(
            {"cue": 1, "food": 0, "duration": p.get("duration", 1), "act": True}
        )
        if name == "habituation" and p["novel_probe"]:
            schedule.append({"cue": 2, "food": 0, "duration": 1, "act": True})
        w["schedule"] = schedule
    return w


def phase(task: dict, world: dict) -> dict:
    name, p, w = task["name"], task["parameters"], world
    o = {
        key: (
            "none"
            if key == "reward" or name == "delay"
            else "unknown"
            if name == "query"
            else 0
        )
        for key in task["fields"]
    }
    if name == "delay":
        o.pop("cue", None)
    if name == "query":
        o.pop("a", None)
        o.pop("b", None)
    if name in {"conditioning", "habituation"}:
        o.pop("cue", None)
    result = {
        "observation": o,
        "duration": None,
        "act": True,
        "legal_actions": list(range(len(task["actions"]))),
        "done": w["done"],
    }
    if w["done"]:
        result.update(duration=0, act=False, legal_actions=[])
        return result
    if w["stage"] == "feedback":
        o["reward"] = w["feedback"]
        result.update(duration=1, act=False, legal_actions=[])
        return result
    if name == "delay":
        o.pop("cue", None)
        if w["stage"] == "cue" and p["cue_mode"] != "blank":
            o["cue"] = w["target"] ^ int(p["cue_mode"] == "inverted")
        result.update(
            duration=p["cue_updates"]
            if w["stage"] == "cue"
            else p["delay"]
            if w["stage"] == "rest"
            else 0,
            act=w["stage"] == "choice",
        )
    elif name == "query":
        if w["latest"] is not None:
            key = "ab"[w["latest"]]
            o[key] = w["features"][w["latest"]]
        result["legal_actions"] = (
            [i for i in range(2) if w["known"][i] == "unknown"]
            if w["queries"] < p["max_queries"]
            else []
        ) + [2, 3, 4, 5]
    elif name == "reversal":
        o["cue"] = w["cue"]
    elif name in {"conditioning", "habituation"}:
        step = w["schedule"][w["trial"]]
        o.update(food=step["food"])
        if step["cue"]:
            o["cue"] = step["cue"]
        result.update(duration=step["duration"], act=step["act"])
    if not result["act"]:
        result["legal_actions"] = []
    return result


def transition(task: dict, world: dict, action, rng) -> tuple:
    current = phase(task, world)
    if current["done"]:
        raise ValueError("episode is done")
    if (
        current["act"]
        and (type(action) is not int or action not in current["legal_actions"])
    ) or (not current["act"] and action is not None):
        raise ValueError("action does not match phase")
    w = deepcopy(world)
    name, p = task["name"], task["parameters"]
    if w["stage"] == "feedback":
        w["stage"] = "choice"
        w["trial"] += 1
        w["feedback"] = None
        limit = len(w["schedule"]) if "schedule" in w else p.get("trials", 1)
        w["done"] = w["trial"] >= limit
        if name == "reversal" and not w["done"]:
            w["cue"] = int(rng.integers(2))
        return w, None, {}
    if name == "delay" and w["stage"] in {"cue", "rest"}:
        w["stage"] = "rest" if w["stage"] == "cue" else "choice"
        return w, None, {}
    if name == "query" and action < 2:
        w["known"][action] = w["features"][action]
        w["queries"] += 1
        w["latest"] = action
        return w, None, {"query": action}
    if not current["act"]:
        w["trial"] += 1
        return w, None, {}
    if name in {"bandit", "advisers"}:
        reward = float(rng.random() < w["probabilities"][action])
    elif name == "boxes" and p["mode"] == "random":
        reward = float(rng.random() < (2 / 3 if action == w["target"] else 1 / 3))
    elif name == "query":
        reward = float(action - 2 == (2 * w["features"][0] + w["features"][1]))
    elif name == "reversal":
        reward = float(
            action == (w["cue"] ^ w["target"] ^ int(w["trial"] >= p["trials"] // 2))
        )
    elif name in {"conditioning", "habituation"}:
        # Probe score records response presence (conditioning) or suppression
        # (habituation); trial position and cumulative score stay hidden.
        reward = float(action == (1 if name == "conditioning" else 0))
    else:
        reward = float(action == w["target"])
    w["feedback"] = reward
    w["score"] += reward
    w["stage"] = "feedback"
    info = {"trial": w["trial"]}
    if name == "habituation":
        response_index = len(w["responses"])
        w["responses"].append(action)
        if p["feedback"] == "none":
            w["feedback"] = "none"
        elif p["feedback"] == "yoked":
            w["feedback"] = p["yoked_rewards"][response_index]
        info.update(
            response=action,
            initial_response=w["responses"][0],
            recovery_response=action if response_index == p["repetitions"] else None,
            novel_response=action if response_index == p["repetitions"] + 1 else None,
            feedback_control=p["feedback"],
            objective="response suppression; constant-zero policy reaches maximum reward",
        )
    return w, reward, info


class TaskInteraction:
    """Expose a resolved episodic task through the situated interaction protocol."""

    actors = ("cell",)

    def __init__(self, task):
        self.task = deepcopy(task)
        self.fields = deepcopy(task["fields"])
        self.actions = list(task["actions"])

    def initial_world(self, rng):
        return episode(self.task, rng)

    def stage(self, world):
        from ..situated import Stage

        current = phase(self.task, world)
        return Stage(
            "task", self.actors if current["act"] else (), duration=current["duration"]
        )

    def observations(self, world):
        return {"cell": phase(self.task, world)["observation"]}

    def legal_actions(self, world, actor):
        if actor != "cell":
            raise ValueError("unknown actor")
        return [self.actions[i] for i in phase(self.task, world)["legal_actions"]]

    def transition(self, world, actions, rng):
        from ..situated import StageResult

        current = phase(self.task, world)
        expected = {"cell"} if current["act"] else set()
        if set(actions) != expected:
            raise ValueError("actions do not match active actors")
        action = self.actions.index(actions["cell"]) if current["act"] else None
        updated, reward, info = transition(self.task, world, action, rng)
        return StageResult(
            updated,
            feedback={"cell": {"score": reward}} if reward is not None else {},
            terminated=phase(self.task, updated)["done"],
            info=info,
        )
