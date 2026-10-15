"""Bounded acquisition assay for initial disposition mass and plan representation.

Run after committing the implementation, with an installed package or PYTHONPATH=src:
    python examples/adaptive_dispositions_study.py --output /path/to/new-study
The script writes its protocol before training and refuses to overwrite an output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from self_assembling_games import AdaptiveProfile, build_adaptive


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def conditions():
    result = []
    for scale in (0.1, 1.0, 10.0):
        for name, method, weight, matching in (
            (
                "conditional",
                "reinforcement.conditional.R1",
                scale,
                "reference conditional tables",
            ),
            (
                "plan_role_mass",
                "reinforcement.strategy.R1",
                scale,
                "matched total mass per role to conditional; table mass doubles",
            ),
            (
                "plan_table_mass",
                "reinforcement.strategy.R1",
                scale / 2,
                "matched mass per adaptive table to conditional; role mass halves",
            ),
        ):
            result.append(
                {
                    "id": f"{name}-{scale:g}",
                    "family": name,
                    "scale": scale,
                    "method": method,
                    "initial_weight": weight,
                    "matching": matching,
                }
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--plays", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args(argv)
    if min(args.seeds, args.plays, args.checkpoint_every) <= 0:
        parser.error("seeds, plays and checkpoint-every must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0,1]")
    commit = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.root), "status", "--porcelain"], text=True
    ).strip()
    arms = conditions()
    checkpoints = sorted(
        set(range(0, args.plays + 1, args.checkpoint_every)) | {args.plays}
    )
    protocol = {
        "study": "binary_signalling_acquisition_disposition_mass_v1",
        "source_commit": commit,
        "source_root": str(args.root.resolve()),
        "source_dirty": bool(dirty),
        "source_status": dirty,
        "hypothesis": "At equal initial uniform encounter policies, lower initial mass changes adaptation speed under additive R1 rewards.",
        "representation_expectation": "Conditional and complete-plan updates may produce different acquisition trajectories under matched role mass; matching table mass provides a second gain control.",
        "interpretation": "Role-matched arms equalize total initial role mass. Table-matched arms equalize each adaptive table's mass and have unequal total role mass. Neither comparison isolates representation from all update-granularity effects.",
        "source_model": "SK-SIG",
        "game_parameters": {"states": 2, "state_probabilities": [0.5, 0.5]},
        "initial_policy": "Independent uniform conditional actions; uniform complete plans induce the same encounter distribution and initial success probability 0.5.",
        "training": "Both sender and receiver learn; success pays each role one; no interventions or frozen components.",
        "seeds": list(range(args.seeds)),
        "seed_pairing": "Independent runs within each arm; common seed labels across arms, without a common-random-number coupling guarantee.",
        "plays": args.plays,
        "checkpoints": checkpoints,
        "threshold": args.threshold,
        "crossing_definition": "First checkpoint with exact frozen-policy success >= threshold; null means unobserved within the encounter budget. Crossing can subsequently reverse.",
        "measurements": [
            "exact endpoint success probability",
            "cumulative training encounter accuracy",
            "first observed threshold crossing in encounters",
        ],
        "summary": "Means and sample standard deviations across seeds; crossing count and median among observed crossings; no confirmatory significance test.",
        "scope": "Acquisition in a stationary two-state game; exploratory constructed protocol.",
        "conditions": arms,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    raw = []
    with (args.output / "runs.jsonl").open("w") as stream:
        for arm in arms:
            for seed in protocol["seeds"]:
                profile = AdaptiveProfile(
                    arm["id"],
                    "SK-SIG",
                    protocol["game_parameters"],
                    method=arm["method"],
                    protocol={
                        "initial_weight": arm["initial_weight"],
                        "plays": args.plays,
                        "window": 100,
                    },
                    kind="deliberate_variant",
                    assumptions=["Constructed stationary acquisition assay"],
                )
                simulation = build_adaptive(profile, seed=seed)
                initial_mass = {
                    k: float(w.sum()) for k, w in simulation.weights.items()
                }
                trajectory, crossing = [], None
                for checkpoint in checkpoints:
                    simulation.run(plays=checkpoint - simulation.plays)
                    final = simulation.evaluate()["success_probability"]
                    trajectory.append(
                        {
                            "plays": checkpoint,
                            "final_policy_success_probability": final,
                            "cumulative_accuracy": simulation.successes / checkpoint
                            if checkpoint
                            else None,
                        }
                    )
                    if crossing is None and final >= args.threshold:
                        crossing = checkpoint
                record = {
                    "condition": arm["id"],
                    "seed": seed,
                    "profile": profile.to_dict(),
                    "initial_table_mass": initial_mass,
                    "initial_total_role_mass": {
                        role: sum(
                            mass
                            for key, mass in initial_mass.items()
                            if key == role or key.startswith(role + "/")
                        )
                        for role in simulation.game.roles
                    },
                    "trajectory": trajectory,
                    "threshold_crossing_plays": crossing,
                    "final_policy_success_probability": trajectory[-1][
                        "final_policy_success_probability"
                    ],
                    "cumulative_accuracy": trajectory[-1]["cumulative_accuracy"],
                }
                raw.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
    summary = []
    for arm in arms:
        records = [r for r in raw if r["condition"] == arm["id"]]
        crossings = [
            r["threshold_crossing_plays"]
            for r in records
            if r["threshold_crossing_plays"] is not None
        ]
        item = {
            "condition": arm["id"],
            "runs": len(records),
            "crossed_runs": len(crossings),
            "median_crossing_plays_observed": float(np.median(crossings))
            if crossings
            else None,
        }
        for metric in ("final_policy_success_probability", "cumulative_accuracy"):
            values = [r[metric] for r in records]
            item[metric] = {
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        summary.append(item)
    write_json(args.output / "summary.json", summary)
    print(f"Saved protocol, {len(raw)} seed runs and summaries to {args.output}")


if __name__ == "__main__":
    main()
