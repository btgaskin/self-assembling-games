"""Box interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows
from .task import encounter as encounter


def initialize(model_instance: ModelInstance) -> None:
    rt, rule = model_instance.runtime, model_instance.learner
    rows(rt, "choice", ["start"], [0, 1], rule)
    model_instance.world["special"] = rt.uniform("special_box", 2)


def validate(profile):

    p = profile.parameters
    if profile.model == "BOX" and p["placement"] not in {"fixed", "random"}:
        raise ValueError("BOX placement must be fixed or random")
