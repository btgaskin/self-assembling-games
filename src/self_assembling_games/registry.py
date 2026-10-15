"""Explicit profiles and a complete, source-linked catalogue."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

RULES = {
    "R1",
    "R2",
    "SIGNED",
    "R3",
    "R4",
    "R6",
    "WSLS",
    "WSLR",
    "STRATEGY_WSLR",
    "BOUNDED",
    "DISCOUNT",
}


def catalogue() -> list[dict[str, Any]]:
    return json.loads(files(__package__).joinpath("data/catalogue.json").read_text())


def profiles() -> list[dict[str, Any]]:
    from .conflict_profiles import conflict_profiles

    return (
        json.loads(files(__package__).joinpath("data/profiles.json").read_text())
        + conflict_profiles()
    )


def reproduction_protocols() -> list[dict[str, Any]]:
    return json.loads(
        files(__package__).joinpath("data/reproductions.json").read_text()
    )


@dataclass
class Profile:
    id: str
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    learner: dict[str, Any] = field(default_factory=lambda: {"rule": "R1"})
    kind: str = "demonstration"
    plays: int = 10_000
    source: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self))

    def validate(self) -> None:
        from .definitions import game_definition

        game_definition(self.model, "barrett.conditional").validate(self)


def validate_distribution(values: list[float], count: int) -> None:
    import math

    if (
        len(values) != count
        or any(not math.isfinite(x) or x < 0 for x in values)
        or not math.isclose(sum(values), 1, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("Invalid probability distribution")


def load_profile(identifier: str | Path | dict[str, Any] | Profile) -> Profile:
    from .adaptive_runtime import is_adaptive_profile

    if is_adaptive_profile(identifier):
        from .finite_profiles import load_adaptive_profile

        return load_adaptive_profile(identifier)
    if isinstance(identifier, Profile):
        profile = copy.deepcopy(identifier)
    elif isinstance(identifier, dict):
        profile = Profile(**copy.deepcopy(identifier))
    else:
        path = Path(identifier)
        if path.is_file():
            profile = Profile(**json.loads(path.read_text()))
        else:
            matches = [p for p in profiles() if p["id"] == str(identifier)]
            if not matches:
                raise ValueError(f"Unknown profile {identifier}; use sg list")
            profile = Profile(**matches[0])
    profile.validate()
    return profile
