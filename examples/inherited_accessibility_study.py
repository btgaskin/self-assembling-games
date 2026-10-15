"""Bounded constructed-function assays; emits protocol, raw runs and summaries.

Run after committing the reviewed implementation:
  PYTHONPATH=src python examples/inherited_accessibility_study.py --output results.json
Add --wright-fisher for 16 seeds, 128 haploids and 100 generations per condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from self_assembling_games import build, load_adaptive_profile


GENERATIONS = 100
SEEDS = tuple(range(16))
STAGES = ("before", "after_selection", "after_recombination", "expected_offspring")


def revision():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "working_tree_status": git("status", "--porcelain"),
            "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}


def transfer_profile(target, encoding, finite=False):
    suffix = "wf" if finite else "deterministic"
    profile = load_adaptive_profile(f"ml-transfer.{suffix}.demo")
    profile.parameters["target"] = list(target)
    profile.genetics["transmission"]["recombination"] = {"kind": "clonal"}
    if encoding == "coupled":
        profile.genetics["expression"]["plan_indices"]["adapter2"] = [0, 1, 1, 0]
    if finite:
        profile.genetics["population_size"] = 128
        profile.protocol["initialization"] = {"kind": "explicit_counts", "counts": [128, 0, 0, 0]}
    else:
        profile.protocol["initialization"] = {"kind": "frequencies", "frequencies": [1, 0, 0, 0]}
    return profile


def transfer_run(target, encoding, seed, finite=False):
    profile = transfer_profile(target, encoding, finite)
    simulation = build(profile, seed=seed, protocol={"generations": GENERATIONS})
    index = 2 * target[0] + (target[1] if encoding == "separable" else target[0] ^ target[1])

    def measurement():
        row = {"generation": simulation.generation,
               "frequencies": simulation.frequencies.tolist(),
               "target_genotype_frequency": float(simulation.frequencies[index]),
               "functional_success_probability": simulation.evaluate()["success_probability"]}
        if finite:
            row["counts"] = simulation.counts.tolist()
        return row

    trajectory = [measurement()]
    first = None
    for _ in range(GENERATIONS):
        record = simulation.step()
        if first is None:
            first = record
        trajectory.append(measurement())
    return {"target": list(target), "encoding": encoding, "seed": seed,
            "method": profile.method, "profile": profile.to_dict(),
            "target_genotype_index": index, "first_generation": first,
            "first_generation_expected_target_probability": first["expected_offspring"][index],
            "trajectory": trajectory}


def stage_assays():
    coordination = []
    for fraction in (0, 0.05, 0.5):
        profile = load_adaptive_profile("ml-coord.deterministic.demo")
        profile.protocol["initialization"] = {"kind": "frequencies", "frequencies": [0.25] * 4}
        profile.genetics["transmission"]["recombination"] = {
            "kind": "independent_intervals", "fractions": [fraction]}
        record = build(profile).step()
        coordination.append({"recombination_fraction": fraction, "profile": profile.to_dict(),
                             "first_generation": record,
                             "stage_success": {stage: record[stage][0] + record[stage][3]
                                               for stage in STAGES}})
    profile = load_adaptive_profile("ml-compose.deterministic.demo")
    profile.protocol["initialization"] = {"kind": "frequencies", "frequencies":
                                         [0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0]}
    profile.genetics["transmission"]["recombination"] = {
        "kind": "single_boundary_exchange", "exchange_probability": 0.5}
    record = build(profile).step()
    composition = {"profile": profile.to_dict(), "first_generation": record,
                   "stage_success": {stage: sum(record[stage][i] for i in (0, 3, 5, 6))
                                     for stage in STAGES}}
    return {"coordination": coordination, "composition": composition}


def summarize(runs):
    summaries = []
    for target in ((1, 0), (1, 1)):
        for encoding in ("separable", "coupled"):
            selected = [r for r in runs if r["target"] == list(target) and r["encoding"] == encoding]
            row = {"target": list(target), "encoding": encoding, "runs": len(selected)}
            for metric in ("target_genotype_frequency", "functional_success_probability"):
                values = np.array([r["trajectory"][-1][metric] for r in selected])
                row[metric] = {"mean": float(values.mean()), "minimum": float(values.min()),
                               "maximum": float(values.max()),
                               "quantiles_025_050_075": np.quantile(values, [0.25, 0.5, 0.75]).tolist()}
            row["first_generation_expected_target_probability"] = selected[0]["first_generation_expected_target_probability"]
            summaries.append(row)
    return summaries


def study(include_wright_fisher=False):
    protocol = {
        "scope": "Constructed function assays with supplied NAND template and preformed interfaces",
        "generations": GENERATIONS, "targets": [[1, 0], [1, 1]],
        "founder": "genotype 00; same expressed founder for both encodings",
        "encoding_comparison": "Change adapter2 expression while retaining independent bit mutation and clonal transmission",
        "mutation": "Symmetric independent bit mutation 0.01 per locus per generation",
        "transfer_recombination": "clonal in all transfer conditions",
        "fitness": {"background": 1, "scale": 1},
        "deterministic": "Finite genotype expectations and floating-point recursion, generations 0 through 100",
        "wright_fisher": {"enabled": include_wright_fisher, "seeds": list(SEEDS), "population_size": 128},
        "outcomes": ["first-generation expected target genotype probability", "target genotype frequency", "functional success probability"],
        "stop_rule": "Exactly 100 generations per run; no adaptive stopping or parameter search",
        "coordinate_null": {
            "reference": "tests/test_multilocus.py::test_entire_transmission_coordinate_transport_is_a_null",
            "meaning": "A pure relabelling transports expression, initial state and the entire transmission operation",
            "execution": "Existing independent null test; not an additional study condition"},
        "interpretation": "Target-dependent accessibility under inherited encodings; supplied capacities and fixed architecture define the assay",
    }
    deterministic = [transfer_run(target, encoding, 0) for target in ((1, 0), (1, 1))
                     for encoding in ("separable", "coupled")]
    finite = ([transfer_run(target, encoding, seed, True) for target in ((1, 0), (1, 1))
               for encoding in ("separable", "coupled") for seed in SEEDS]
              if include_wright_fisher else [])
    return {"schema_version": 1, "revision": revision(), "protocol": protocol,
            "raw_results": {"deterministic": deterministic, "wright_fisher": finite,
                            "stage_assays": stage_assays()},
            "summary": {"deterministic": summarize(deterministic),
                        "wright_fisher": summarize(finite) if finite else []}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wright-fisher", action="store_true")
    args = parser.parse_args()
    result = study(args.wright_fisher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
