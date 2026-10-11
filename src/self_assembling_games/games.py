"""Source-readable interaction functions; observation and credit are explicit."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from .core import Address, Decision, Feedback, Outcome, Runtime, Table

if TYPE_CHECKING:
    from .simulation import Simulation


def rows(
    rt: Runtime,
    module: str,
    contexts: list[Any],
    options: list[Any],
    learner: dict[str, Any],
    *,
    owner: str | None = None,
) -> None:
    for context in contexts:
        rt.add_table(module, context, options, learner, owner=owner)


def fixed(rt: Runtime, module: str, context: Any, option: Any) -> None:
    t = rt.tables[rt.table_id(module, context)]
    t.current = option
    t.frozen = True


def initialize(sim: Simulation) -> None:
    rt, p, model, rule = (
        sim.runtime,
        sim.profile.parameters,
        sim.profile.model,
        sim.profile.learner,
    )
    if model in {"SIG", "CUE", "MAN", "HYB"}:
        rows(rt, "sender", list(range(p["states"])), list(range(p["signals"])), rule)
        rows(rt, "receiver", list(range(p["signals"])), list(range(p["actions"])), rule)
        if model == "CUE":
            for x in range(p["states"]):
                fixed(rt, "sender", x, p["fixed_map"][x])
        if model == "MAN":
            for s in range(p["signals"]):
                fixed(rt, "receiver", s, p["fixed_map"][s])
    elif model in {"COD", "SCOD", "CHANGE"}:
        for module in ["A", "B"]:
            rows(rt, module, list(range(4)), [0, 1], rule)
        pairs = [[a, b] for a in range(2) for b in range(2)]
        rows(rt, "R", pairs, list(range(4)), rule)
        if model == "SCOD":
            rows(rt, "salience", ["start"], ["A", "B", "AB"], rule)
            for module in ["RA", "RB"]:
                rows(rt, module, [0, 1], list(range(4)), rule)
    elif model == "NAND":
        nand_rows(rt, "", rule)
    elif model == "HIER":
        for module in ["A", "B"]:
            rows(rt, module, list(range(4)), [0, 1], rule)
        rows(rt, "ES", [0, 1, 2], ["A", "B", "AB"], rule)
        rows(rt, "ER", ["A", "B", "AB"], [0, 1, 2], rule)
        rows(
            rt, "R", [[a, b] for a in range(2) for b in range(2)], list(range(4)), rule
        )
    elif model in {"INV", "ORD", "ADD"}:
        modules = ["sender"] if model == "INV" else ["A", "B"]
        for module in modules:
            rows(rt, module, list(range(p["n"])), ["mutator"], {"rule": "R1"})
    elif model in {"TR-LOG", "COMP-NAND"}:
        if model == "TR-LOG":
            install_nand(sim, "template:", "template")
            for module in ["TA", "TB"]:
                rows(rt, module, [0, 1], [0, 1], rule)
        else:
            for name in ["first", "second"]:
                install_nand(sim, name + ":", name)
            rows(rt, "P", ["start"], ["Z", "first"], rule)
            if p["selectors"] == 2:
                rows(rt, "Q", ["start"], ["Q", "first"], rule)
    elif model in {"TR-ORD", "TR-PART", "ADD-COMP"}:
        if model == "ADD-COMP":
            if p["template_mode"] == "artifact":
                install_artifact(sim, "addition", "addition:", "ADD")
                install_artifact(sim, "comparison", "ordering:", "ORD")
            rows(rt, "translation", list(range(11)), list(range(11)), rule)
        else:
            if p["template_mode"] == "artifact":
                install_artifact(sim, "ordering", "ordering:", "ORD")
            for module in ["TA", "TB"]:
                rows(rt, module, list(range(p["n"])), list(range(p["n"])), rule)
    elif model in {"DIAL", "DIAL-restricted", "DIAL-costly", "ASK"}:
        answers = p["answers"]
        materials = 2 if model == "ASK" else 4
        questions = ["Q"] if model == "ASK" else ["Q0", "Q1"]
        choices = list(range(materials)) + questions
        rows(rt, "assistant", ["start"], choices, rule)
        rows(rt, "reply1", list(range(answers)), choices, rule)
        if model != "ASK":
            rows(
                rt,
                "reply2",
                [[a, b] for a in range(answers) for b in range(answers)],
                choices,
                rule,
            )
        rows(
            rt,
            "builder",
            [[q, m] for q in questions for m in range(materials)],
            list(range(answers)),
            rule,
        )
    elif model in {"VIS", "STAG", "FFSR", "INQ"}:
        n = p["agents"]
        for i in range(n):
            options = (
                ["nature"] + list(range(n))
                if model == "INQ"
                else [j for j in range(n) if j != i]
            )
            rows(rt, f"visit:{i}", ["start"], options, rule, owner=f"agent:{i}")
            if model == "FFSR":
                rows(
                    rt,
                    f"strategy:{i}",
                    ["start"],
                    list(range(16)),
                    rule,
                    owner=f"agent:{i}",
                )
        if model == "INQ":
            sim.world["beliefs"] = [False] * n
            for field in ["epistemic", "social"]:
                if field in p:
                    sim.world[field] = list(p[field])
                elif field == "social" and p.get("perfect_communication"):
                    sim.world[field] = [1.0] * n
                else:
                    # Continuous initialization is saved in the checkpoint; evaluations condition on it.
                    sim.world[field] = rt.rng.random(n).tolist()
                    rt.emit("chance", label=field, values=sim.world[field])
    elif model == "BOX":
        rows(rt, "choice", ["start"], [0, 1], rule)
        sim.world["special"] = rt.uniform("special_box", 2)
    elif model in {"BAND", "META-ORACLE", "META-RULE", "MH"}:
        count = (
            len(p["arms"])
            if model == "BAND"
            else len(p["reliabilities"])
            if model == "META-ORACLE"
            else len(p["rules"])
            if model == "META-RULE"
            else 2
        )
        rows(rt, "choice", ["start"], list(range(count)), rule)
    elif model in {"ATT-R", "ATT-B"}:
        receiver_features = p["receiver_features"]
        contexts = (
            [0, 1]
            if model == "ATT-R" or p["binding"] == "value"
            else [
                [source, value]
                for source in range(p["sender_features"])
                for value in [0, 1]
            ]
        )
        rows(rt, "sender", contexts, [0, 1], rule)
        if model == "ATT-B":
            rows(
                rt,
                "sender_attention",
                ["start"],
                list(range(p["sender_features"])),
                rule,
            )
        rows(rt, "receiver_attention", ["start"], list(range(receiver_features)), rule)
        rows(
            rt,
            "receiver",
            [[f, v] for f in range(receiver_features) for v in [0, 1]],
            [0, 1],
            rule,
        )
    else:
        raise ValueError(f"No operational game for {model}")


def nand_rows(rt: Runtime, prefix: str, learner: dict[str, Any]) -> None:
    for module in ["A", "B"]:
        rows(rt, prefix + module, [0, 1], [0, 1], learner)
    rows(rt, prefix + "R", [[a, b] for a in [0, 1] for b in [0, 1]], [0, 1], learner)


def install_artifact(sim: Simulation, name: str, prefix: str, expected: str) -> None:
    from .simulation import read_snapshot

    snapshot = read_snapshot(sim.profile.parameters["templates"][name])
    if snapshot["profile"]["model"] != expected:
        raise ValueError(f"Template {name} must be a {expected} artifact")
    if expected == "NAND" and snapshot["profile"]["parameters"].get(
        "truth_table", [1, 1, 1, 0]
    ) != [1, 1, 1, 0]:
        raise ValueError(
            "A NAND template must have been trained for the NAND truth function"
        )
    if expected in {"ORD", "ADD"}:
        tables = snapshot["runtime"]["tables"]
        for record in tables.values():
            if record["module"] in {"A", "B"} and "mutator" in record["options"]:
                raise ValueError(
                    "Frozen vocabulary needs an explicit export_template mutator policy"
                )
        if snapshot["profile"]["parameters"]["n"] != (
            sim.profile.parameters.get("n", 11) if expected == "ORD" else 6
        ):
            raise ValueError("Template stimulus domain differs from the adapter domain")
        import itertools

        vocab = {
            m: {
                option
                for record in tables.values()
                if record["module"] == m
                for option in record["options"]
            }
            for m in ["A", "B"]
        }
        if any(
            sim.runtime.table_id("R", [a, b]) not in tables
            for a, b in itertools.product(vocab["A"], vocab["B"])
        ):
            raise ValueError(
                "Frozen vocabulary lacks a receiver row for an available signal pair"
            )
    for old in snapshot["runtime"]["tables"].values():
        record = copy.deepcopy(old)
        record["module"] = prefix + record["module"]
        record["frozen"] = True
        table = Table.from_dict(record)
        sim.runtime.tables[sim.runtime.table_id(table.module, table.context)] = table
        sim.runtime.owners[table.module] = prefix.rstrip(":")
        sim.runtime.frozen_modules.add(table.module)
    sim.lineage[name] = {
        "path": str(sim.profile.parameters["templates"][name]),
        "profile": snapshot["profile"],
        "metrics": snapshot["metrics"],
        "seed": snapshot["seed"],
        "play": snapshot["runtime"]["clocks"]["play"],
        "configuration_hash": snapshot["configuration_hash"],
    }
    sim.lineage[name]["artifact_hash"] = sim_hash(snapshot)
    sim.lineage[name]["export_transforms"] = snapshot["lineage"].get(
        "vocabulary_export"
    )


def sim_hash(snapshot: dict[str, Any]) -> str:
    from .simulation import digest

    return digest(snapshot)


def install_nand(sim: Simulation, prefix: str, name: str) -> None:
    if sim.profile.parameters["template_mode"] == "artifact":
        install_artifact(sim, name, prefix, "NAND")
        return
    rt = sim.runtime
    nand_rows(rt, prefix, {"rule": "R1"})
    for x in [0, 1]:
        for module in ["A", "B"]:
            fixed(rt, prefix + module, x, x)
    for a in [0, 1]:
        for b in [0, 1]:
            fixed(rt, prefix + "R", [a, b], 1 - a * b)
    sim.lineage[name] = {"kind": "constructed_perfect_fixture", "function": "NAND"}


def nand_call(rt: Runtime, prefix: str, a: int, b: int) -> tuple[int, list[Decision]]:
    da = rt.decide(prefix + "A", a, a)
    db = rt.decide(prefix + "B", b, b)
    dr = rt.decide(prefix + "R", [da.option, db.option], [da.option, db.option])
    return dr.option, [da, db, dr]


def ordering_call(rt: Runtime, a: int, b: int) -> int:
    da = rt.decide("ordering:A", a, a)
    db = rt.decide("ordering:B", b, b)
    return rt.decide(
        "ordering:R", [da.option, db.option], [da.option, db.option]
    ).option


def relation(a: int, b: int) -> int:
    return 0 if a < b else 1 if a > b else 2


def play(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    if model in {"SIG", "CUE", "MAN", "HYB"}:
        x = rt.chance(
            "nature", p.get("state_probabilities", [1 / p["states"]] * p["states"])
        )
        sender = rt.decide("sender", x, x)
        receiver = rt.decide("receiver", sender.option, sender.option)
        reward = (
            p["payoffs"][x][receiver.option]
            if "payoffs" in p
            else int(x == receiver.option)
        )
        success = (
            bool(p["success_matrix"][x][receiver.option])
            if "success_matrix" in p
            else x == receiver.option
        )
        outcome = Outcome(success, reward)
        if sim.profile.learner["rule"] == "STRATEGY_WSLR":
            if not outcome.success and rt.training:
                for identity, table in rt.tables.items():
                    if not table.frozen:
                        table.current = table.options[
                            rt.uniform("new_plan:" + identity, len(table.options))
                        ]
                        rt.emit("assignment", table=identity, option=table.current)
        else:
            rt.credit([sender, receiver], outcome)
        return outcome
    if model in {"COD", "SCOD", "CHANGE"}:
        x = rt.uniform("nature", 4)
        a = rt.decide("A", x, x)
        b = rt.decide("B", x, x)
        choices = [a, b]
        if model == "SCOD":
            selected_attention = rt.decide("salience", "start", "start")
            choices.append(selected_attention)
            module, context = (
                ("R", [a.option, b.option])
                if selected_attention.option == "AB"
                else (
                    "R" + selected_attention.option,
                    a.option if selected_attention.option == "A" else b.option,
                )
            )
        else:
            module, context = "R", [a.option, b.option]
        action = rt.decide(module, context, context)
        choices.append(action)
        target = (
            p["new_mapping"][x]
            if model == "CHANGE" and rt.clocks["play"] >= p["change_at"]
            else x
        )
        outcome = Outcome(action.option == target, float(action.option == target))
        rt.credit(choices, outcome)
        return outcome
    if model == "NAND":
        a = rt.uniform("input_A", 2)
        b = rt.uniform("input_B", 2)
        output, choices = nand_call(rt, "", a, b)
        target = p.get("truth_table", [1, 1, 1, 0])[2 * a + b]
        outcome = Outcome(output == target, float(output == target))
        rt.credit(choices, outcome)
        return outcome
    if model == "HIER":
        x = rt.uniform("properties", 4)
        context = rt.uniform("context", 3)
        executive = rt.decide("ES", context, context)
        active = executive.option
        choices = [executive]
        signals = {}
        for module in ["A", "B"]:
            if module in active:
                d = rt.decide(module, x, x)
                choices.append(d)
                signals[module] = d.option
        pair = [
            signals["A"] if "A" in signals else rt.uniform("missing_A", 2),
            signals["B"] if "B" in signals else rt.uniform("missing_B", 2),
        ]
        action = rt.decide("R", pair, {"expression": signals})
        choices.append(action)
        interpretation = rt.decide("ER", active, active)
        choices.append(interpretation)
        projection = [lambda y: y // 2, lambda y: y % 2, lambda y: y][context]
        success = interpretation.option == context and projection(
            action.option
        ) == projection(x)
        outcome = Outcome(
            success,
            (p["complex_reward"] if context == 2 else p["simple_reward"])
            if success
            else 0.0,
            p["term_cost"] * len(signals),
            terms=len(signals),
        )
        rt.credit(choices, outcome)
        return outcome
    if model in {"INV", "ORD", "ADD"}:
        return invention(sim)
    if model in {"TR-LOG", "COMP-NAND", "TR-ORD", "TR-PART", "ADD-COMP"}:
        return composition(sim)
    if model in {"DIAL", "DIAL-restricted", "DIAL-costly", "ASK"}:
        return dialogue(sim)
    if model in {"VIS", "STAG", "FFSR"}:
        return network(sim)
    if model == "INQ":
        return inquiry(sim)
    if model in {"BOX", "BAND", "META-ORACLE", "META-RULE", "MH"}:
        return task(sim)
    if model in {"ATT-R", "ATT-B"}:
        return attention(sim)
    raise ValueError(model)


def invention(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    modules = ["sender"] if model == "INV" else ["A", "B"]
    inputs = [
        rt.chance("input:" + m, p.get("state_probabilities", [1 / p["n"]] * p["n"]))
        for m in modules
    ]
    draws = [rt.decide(m, x, x) for m, x in zip(modules, inputs)]
    proposed = [d.option == "mutator" for d in draws]
    signals = [
        rt.fresh(d.module) if new else d.option for d, new in zip(draws, proposed)
    ]
    context = signals[0] if model == "INV" else signals
    identity = rt.table_id("R", context)
    created = identity not in rt.tables
    count = p["n"] if model == "INV" else 3 if model == "ORD" else 2 * p["n"] - 1
    if created:
        if len(rt.tables) >= p["max_tables"]:
            return Outcome(
                False, status="resource_limit", details={"limit": "max_tables"}
            )
        rt.add_table("R", context, list(range(count)), {"rule": "R1"})
    receiver = rt.decide("R", context, context)
    target = (
        inputs[0]
        if model == "INV"
        else relation(*inputs)
        if model == "ORD"
        else sum(inputs)
    )
    success = receiver.option == target
    outcome = Outcome(success, float(success))
    if rt.training:
        if success:
            for d, new, signal in zip(draws, proposed, signals):
                if new:
                    for address, table in list(rt.tables.items()):
                        if table.module == d.module:
                            rt.insert(address, signal)
                    rt.feedback(
                        Feedback(Address(d.table, signal), True, 1.0, (d.occurrence,))
                    )
                else:
                    rt.credit([d], outcome)
            rt.credit([receiver], outcome)
            rt.emit("commit_invention", signals=signals)
        elif (
            created
            and any(proposed)
            and (model == "INV" or p["failed_row"] == "discard")
        ):
            rt.remove(identity)
            rt.emit("rollback_invention", signals=signals)
        if (
            model == "INV"
            and p.get("forgetting")
            and (rt.clocks["play"] + 1) % p["forgetting"]["every"] == 0
        ):
            forget_signal(rt, p["forgetting"])
    return outcome


def forget_signal(rt: Runtime, config: dict[str, Any]) -> None:
    """An explicit type/ball forgetting variant with declared extinction semantics."""
    urns = [(identity, t) for identity, t in rt.tables.items() if t.module == "sender"]
    identity, table = urns[rt.uniform("forget_urn", len(urns))]
    eligible = [
        i
        for i, option in enumerate(table.options)
        if not config["protect_mutator"] or option != "mutator"
    ]
    if not eligible:
        return
    probabilities = (
        [1 / len(eligible)] * len(eligible)
        if config["mode"] == "type"
        else (table.weights[eligible] / table.weights[eligible].sum()).tolist()
    )
    index = eligible[rt.chance("forget_option", probabilities)]
    option = table.options[index]
    if table.weights[index] <= 1 and len(table.options) == 1:
        rt.emit("forget_skipped", reason="protect_empty_urn")
        return
    if table.weights[index] <= 1:
        rt.delete_option(identity, option)
        represented = any(option in t.options for _, t in urns)
        receiver = rt.table_id("R", option)
        if (
            not represented
            and config["cleanup"] == "remove_orphan_receiver"
            and receiver in rt.tables
        ):
            rt.remove(receiver)
    else:
        table.weights[index] -= 1
        table.validate()
    rt.emit("forget_signal", table=identity, option=option, mode=config["mode"])


def composition(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    if model == "TR-LOG":
        a, b = rt.uniform("input_A", 2), rt.uniform("input_B", 2)
        da, db = rt.decide("TA", a, a), rt.decide("TB", b, b)
        output, _ = nand_call(rt, "template:", da.option, db.option)
        target = p["truth_table"][a * 2 + b]
        outcome = Outcome(output == target, float(output == target))
        rt.credit([da, db], outcome)
        return outcome
    if model == "COMP-NAND":
        x, y, q, z = [rt.uniform(name, 2) for name in ["X", "Y", "Q", "Z"]]
        first = rt.invoke("first", lambda: nand_call(rt, "first:", x, y)[0])
        selector = rt.decide("P", "start", "start")
        choices = [selector]
        a = z if selector.option == "Z" else first
        b = q
        if p["selectors"] == 2:
            second_selector = rt.decide("Q", "start", "start")
            choices.append(second_selector)
            b = q if second_selector.option == "Q" else first
        output = rt.invoke("second", lambda: nand_call(rt, "second:", a, b)[0])
        target = 1 - (1 - x * y) * q
        outcome = Outcome(output == target, float(output == target))
        rt.credit(choices, outcome)
        return outcome
    if model in {"TR-ORD", "TR-PART"}:
        if model == "TR-PART":
            a = rt.uniform("adjacent_pair", p["n"] - 1)
            b = a + 1
            if rt.uniform("pair_direction", 2):
                a, b = b, a
        else:
            a, b = rt.uniform("new_A", p["n"]), rt.uniform("new_B", p["n"])
        da, db = rt.decide("TA", a, a), rt.decide("TB", b, b)
        output = (
            relation(da.option, db.option)
            if p["template_mode"] == "constructed"
            else ordering_call(rt, da.option, db.option)
        )
        outcome = Outcome(output == relation(a, b), float(output == relation(a, b)))
        rt.credit([da, db], outcome)
        return outcome
    a, b = rt.uniform("addend_A", 6), rt.uniform("addend_B", 6)
    if p["template_mode"] == "constructed":
        summed = a + b
    else:
        da = rt.decide("addition:A", a, a)
        db = rt.decide("addition:B", b, b)
        summed = rt.decide(
            "addition:R", [da.option, db.option], [da.option, db.option]
        ).option
    eligible = [r for r in range(11) if p["ties"] != "exclude" or r != a + b]
    r = eligible[rt.uniform("right_cardinality", len(eligible))]
    translation = rt.decide("translation", summed, summed)
    interpreted = (
        relation(translation.option, r)
        if p["template_mode"] == "constructed"
        else ordering_call(rt, translation.option, r)
    )
    expected = relation(a + b, r)
    tie_answer = 1 if p["ties"] == "left" else 0
    output = tie_answer if interpreted == 2 else interpreted
    target = tie_answer if expected == 2 else expected
    outcome = Outcome(output == target, float(output == target))
    rt.credit([translation], outcome)
    return outcome


def dialogue(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    material = rt.uniform("needed_material", 2 if model == "ASK" else 4)
    choices = []
    answers = []
    exchanges = 0
    module, context = "assistant", "start"
    while True:
        action = rt.decide(module, context, context)
        choices.append(action)
        rt.clocks["turn"] += 1
        if isinstance(action.option, int):
            success = action.option == material
            reward = p["base_reward"] if success else 0.0
            cost = p.get("exchange_cost", 0.0) * exchanges
            if not success and p.get("failure_cost", True) is False:
                cost = 0.0
            outcome = Outcome(success, reward, cost, exchanges=exchanges)
            rt.credit(choices, outcome)
            return outcome
        if exchanges >= p["max_exchanges"]:
            if model in {"DIAL", "DIAL-restricted"}:
                outcome = Outcome(False, exchanges=exchanges)
                rt.credit(choices, outcome)
            else:
                # Engineering censoring supplies no learning update.
                outcome = Outcome(False, exchanges=exchanges, status="truncated")
            return outcome
        reply = rt.decide(
            "builder", [action.option, material], [action.option, material]
        )
        choices.append(reply)
        answers.append(reply.option)
        exchanges += 1
        rt.clocks["turn"] += 1
        if model == "ASK" or exchanges == 1:
            module, context = "reply1", reply.option
        else:
            module, context = "reply2", answers[-2:]
            identity = rt.table_id(module, context)
            if identity not in rt.tables:
                rt.add_table(
                    module, context, list(range(4)) + ["Q0", "Q1"], sim.profile.learner
                )


def plans(plan: int) -> tuple[list[int], list[int]]:
    return [(plan // 4) // 2, (plan // 4) % 2], [(plan % 4) // 2, plan % 2]


def expected_signaling(left: int, right: int) -> float:
    ls, lr = plans(left)
    rs, rr = plans(right)
    return sum(int(rr[ls[x]] == x) + int(lr[rs[x]] == x) for x in [0, 1]) / 4


def network(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    n = p["agents"]
    if not sim.interaction.get("visit_order"):
        sim.interaction["visit_order"] = (
            rt.permutation("visit_order", n)
            if p["schedule"] == "random_permutation"
            else list(range(n))
        )
    visitor = sim.interaction["visit_order"].pop(0)
    visit = rt.decide(f"visit:{visitor}", "start", "start")
    host = visit.option
    if model == "STAG":
        types = p["types"]
        v, h = types[visitor], types[host]
        visitor_payoff = 3.0 if v == "hare" else 4.0 if h == "stag" else 0.0
        host_payoff = 3.0 if h == "hare" else 4.0 if v == "stag" else 0.0
    elif model == "FFSR":
        v = rt.decide(f"strategy:{visitor}", "start", "start")
        h = rt.decide(f"strategy:{host}", "start", "start")
        visitor_payoff = host_payoff = expected_signaling(v.option, h.option)
        rt.credit([v, h], Outcome(visitor_payoff > 0, visitor_payoff))
    else:
        visitor_payoff = host_payoff = p["payoff"]
    outcome = Outcome(
        visitor_payoff > 0,
        visitor_payoff,
        details={"visitor": visitor, "host": host, "host_payoff": host_payoff},
    )
    rt.credit([visit], outcome)
    if model != "FFSR" and p["reciprocal"]:
        rt.feedback(
            Feedback(
                Address(rt.table_id(f"visit:{host}", "start"), visitor),
                host_payoff > 0,
                host_payoff,
                (visit.occurrence,),
            )
        )
    if model == "FFSR" and rt.training:
        for t in rt.tables.values():
            if not t.frozen:
                t.weights[:] = np_maximum_floor(t.weights * p["retention"], p["floor"])
                t.validate()
        rt.emit("forget", scope="all_tables_after_visit", retention=p["retention"])
    if not sim.interaction["visit_order"]:
        rt.clocks["round"] += 1
    return outcome


def np_maximum_floor(weights: Any, floor: float) -> Any:
    import numpy as np

    return np.maximum(weights, floor)


def inquiry(sim: Simulation) -> Outcome:
    rt, p = sim.runtime, sim.profile.parameters
    n = p["agents"]
    if not sim.interaction.get("inquiry_order"):
        sim.interaction["inquiry_order"] = rt.permutation("inquiry_order", n)
    i = sim.interaction["inquiry_order"].pop(0)
    source = rt.decide(f"visit:{i}", "start", "start")
    beliefs = sim.world["beliefs"]
    if source.option == "nature":
        correct = rt.bernoulli("empirical_observation", sim.world["epistemic"][i])
    elif source.option == i:
        correct = beliefs[i]
    else:
        correct = beliefs[source.option] and rt.bernoulli(
            "social_observation", sim.world["social"][i]
        )
    beliefs[i] = bool(correct)
    rt.emit("belief", agent=i, belief=bool(correct))
    outcome = Outcome(
        bool(correct), float(correct), details={"inquirer": i, "source": source.option}
    )
    rt.credit([source], outcome)
    if not sim.interaction["inquiry_order"]:
        rt.clocks["round"] += 1
        if rt.clocks["round"] % p["rounds_per_problem"] == 0:
            rt.clocks["problem"] += 1
            sim.world["beliefs"] = [False] * n
            rt.emit("reset", scope="beliefs", problem=rt.clocks["problem"])
    return outcome


def task(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    choice = rt.decide("choice", "start", "start")
    if model == "BOX":
        rewarded = sim.world["special"]
        if p["placement"] == "random":
            rewarded = (
                rewarded if rt.bernoulli("food_placement", 2 / 3) else 1 - rewarded
            )
        success = choice.option == rewarded
        reward = float(success)
    elif model == "BAND":
        arm = p["arms"][choice.option]
        reward = arm["rewards"][rt.chance("machine_payoff", arm["probabilities"])]
        success = True
    elif model == "META-ORACLE":
        success = rt.bernoulli("oracle_answer", p["reliabilities"][choice.option])
        reward = 1.0 if success else p["failure_return"]
    elif model == "META-RULE":
        from .registry import Profile
        from .simulation import build

        seed = rt.integer("fresh_inner_seed", 2**32)
        selected = p["rules"][choice.option]
        inner = build(
            Profile(
                id="inner",
                model=p["inner_model"],
                parameters=copy.deepcopy(p["inner_parameters"]),
                learner=selected,
                plays=p["inner_plays"],
            ),
            seed=seed,
        )
        inner.run(plays=p["inner_plays"])
        accuracy = inner.metrics.summary()["cumulative_accuracy"]
        success = accuracy > p["threshold"]
        reward = (
            accuracy
            if p["return"] == "accuracy"
            else 1.0
            if success
            else p["failure_return"]
        )
        sim.interaction["last_inner"] = {
            "seed": seed,
            "rule": selected,
            "accuracy": accuracy,
        }
        rt.clocks["problem"] += 1
        rt.emit("inner_task", **sim.interaction["last_inner"])
    else:
        prize = rt.uniform("prize", 3)
        initial = rt.uniform("initial_door", 3)
        eligible = [d for d in range(3) if d != prize and d != initial]
        opened = eligible[rt.uniform("host_goat_door", len(eligible))]
        selected = (
            initial
            if choice.option == 0
            else next(d for d in range(3) if d not in {initial, opened})
        )
        success = selected == prize
        reward = float(success)
    outcome = Outcome(success, reward)
    rt.credit([choice], outcome)
    return outcome


def attention(sim: Simulation) -> Outcome:
    rt, p, model = sim.runtime, sim.profile.parameters, sim.profile.model
    state = rt.uniform("true_state", 2)
    choices = []
    if model == "ATT-B":
        source_values = [state] + [
            rt.uniform("source_distractor", 2) for _ in range(p["sender_features"] - 1)
        ]
        source = rt.decide("sender_attention", "start", "start")
        choices.append(source)
        observed = source_values[source.option]
        observation = {"source": source.option, "value": observed}
        context = (
            [source.option, observed] if p["binding"] == "source_value" else observed
        )
    else:
        context = state
        observation = state
    signal = rt.decide("sender", context, observation)
    choices.append(signal)
    feature_values = [signal.option] + [
        rt.uniform("receiver_distractor", 2) for _ in range(p["receiver_features"] - 1)
    ]
    salience = rt.decide("receiver_attention", "start", "start")
    choices.append(salience)
    value = feature_values[salience.option]
    receiver = rt.decide(
        "receiver",
        [salience.option, value],
        {"feature": salience.option, "value": value},
    )
    choices.append(receiver)
    outcome = Outcome(receiver.option == state, float(receiver.option == state))
    rt.credit(choices, outcome)
    return outcome
