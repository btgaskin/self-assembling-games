"""Validate assay controls without running the scientific acquisition study."""

from pathlib import Path
import runpy

import pytest

from self_assembling_games import AdaptiveProfile, build_adaptive


def test_study_initial_policies_and_mass_controls():
    namespace = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples/adaptive_dispositions_study.py"
        )
    )
    arms = namespace["conditions"]()
    assert len(arms) == 9
    for arm in arms:
        simulation = build_adaptive(
            AdaptiveProfile(
                arm["id"],
                "SK-SIG",
                {"states": 2, "state_probabilities": [0.5, 0.5]},
                method=arm["method"],
                protocol={"initial_weight": arm["initial_weight"]},
            )
        )
        assert simulation.plays == 0
        assert simulation.evaluate()["success_probability"] == pytest.approx(0.5)
        scale = arm["scale"]
        for role in simulation.game.roles:
            tables = [
                w
                for key, w in simulation.weights.items()
                if key == role or key.startswith(role + "/")
            ]
            expected_role_mass = (
                2 * scale if arm["family"] == "plan_table_mass" else 4 * scale
            )
            assert sum(w.sum() for w in tables) == pytest.approx(expected_role_mass)
            expected_table_mass = (
                4 * scale if arm["family"] == "plan_role_mass" else 2 * scale
            )
            assert all(w.sum() == pytest.approx(expected_table_mass) for w in tables)
