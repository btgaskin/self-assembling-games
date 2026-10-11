"""Transfer interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows, install_artifact
from .composition import encounter as encounter


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    if model == "ADD-COMP":
        if p["template_mode"] == "artifact":
            install_artifact(model_instance, "addition", "addition:", "ADD")
            install_artifact(model_instance, "comparison", "ordering:", "ORD")
        rows(rt, "translation", list(range(11)), list(range(11)), rule)
    else:
        if p["template_mode"] == "artifact":
            install_artifact(model_instance, "ordering", "ordering:", "ORD")
        for module in ["TA", "TB"]:
            rows(rt, module, list(range(p["n"])), list(range(p["n"])), rule)


def validate(profile):

    p = profile.parameters
    if profile.model == "ADD-COMP" and p.get("ties") not in {
        "left",
        "right",
        "exclude",
    }:
        raise ValueError("Addition/comparison requires an explicit tie policy")

    if (
        profile.model == "TR-PART"
        and p["pair_sampler"] != "uniform_adjacent_both_directions"
    ):
        raise ValueError(
            "Implemented partial sampler is uniform_adjacent_both_directions"
        )

    if profile.model in {"TR-ORD", "TR-PART", "TR-LOG", "COMP-NAND", "ADD-COMP"}:
        mode = p.get("template_mode")
        if mode not in {"constructed", "artifact"}:
            raise ValueError("template_mode must be constructed or artifact")
        if mode == "constructed" and profile.kind in {
            "book_literal",
            "source_resolved",
        }:
            raise ValueError(
                "Constructed fixtures belong to demonstration or deliberate_variant profiles"
            )
        if mode == "artifact" and not p.get("templates"):
            raise ValueError("Artifact transfer requires templates")
