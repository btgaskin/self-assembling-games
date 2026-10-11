"""Shared scientific source gates and conditional-realisation constraints."""

from .registry import RULES, catalogue


def validate_barrett(profile):
    import math

    entries = {m["id"]: m for m in catalogue()}
    if profile.model not in entries:
        raise ValueError(f"Unknown model {profile.model}")
    entry = entries[profile.model]
    if not entry["implemented"]:
        raise ValueError(
            f"{profile.model} is a catalogue entry awaiting an operational protocol: "
            + ", ".join(entry["required_fields"])
        )
    if profile.kind not in {
        "book_literal",
        "source_resolved",
        "demonstration",
        "deliberate_variant",
    }:
        raise ValueError("Unknown profile kind")
    if (
        profile.kind in {"book_literal", "source_resolved"}
        and entry["source_questions"]
    ):
        raise ValueError(
            f"Source profile {profile.model} needs resolution: "
            + ", ".join(entry["source_questions"])
        )
    missing = [f for f in entry["required_fields"] if f not in profile.parameters]
    if missing:
        raise ValueError("Missing profile fields: " + ", ".join(missing))
    rule = profile.learner.get("rule")
    if rule not in RULES or rule not in entry["learners"]:
        raise ValueError(f"Learner {rule} is unsupported for {profile.model}")
    for name in ["reward", "punishment", "floor", "maximum"]:
        if name in profile.learner and (
            not math.isfinite(profile.learner[name]) or profile.learner[name] < 0
        ):
            raise ValueError(f"Invalid learner parameter {name}")
    for name in [
        "epsilon",
        "forgetting",
        "w_plus",
        "w_minus",
        "error",
        "retention",
    ]:
        if name in profile.learner and not 0 <= profile.learner[name] <= 1:
            raise ValueError(f"Learner {name} must be in [0,1]")
    if not isinstance(profile.plays, int) or profile.plays <= 0:
        raise ValueError("plays must be a positive integer")
    p = profile.parameters
    for field_name in [
        "n",
        "states",
        "signals",
        "actions",
        "phase_length",
        "window",
        "agents",
        "rounds_per_problem",
        "inner_plays",
        "max_exchanges",
        "answers",
        "receiver_features",
        "sender_features",
        "max_tables",
    ]:
        if field_name in p and (
            not isinstance(p[field_name], int) or p[field_name] <= 0
        ):
            raise ValueError(f"{field_name} must be a positive integer")
    if rule == "R3" and (
        not isinstance(profile.learner.get("phase_length"), int)
        or profile.learner["phase_length"] <= 0
    ):
        raise ValueError("R3 requires positive phase_length")
    if rule == "R4":
        missing = {
            "epsilon",
            "floor",
            "forgetting",
            "w_plus",
            "w_minus",
        } - profile.learner.keys()
        if missing:
            raise ValueError("R4 requires " + ", ".join(sorted(missing)))
        if profile.model not in {"BAND", "META-ORACLE"}:
            raise ValueError("R4's source scope is pinned here to single-vector tasks")
    if rule == "BOUNDED" and profile.learner.get("maximum", 0) < 1:
        raise ValueError("BOUNDED requires maximum >= 1")
    if rule == "DISCOUNT" and not 0 < profile.learner.get("retention", 0) <= 1:
        raise ValueError("DISCOUNT requires 0 < retention <= 1")
