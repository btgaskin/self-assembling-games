"""Bounded, unselected Boolean-response survey with exact case assessment.

Run from a clean committed checkout; --output receives protocol and raw results.
Every sampled genome is assessed against all sixteen Boolean targets from the
same four case-response distributions. This is a repertoire survey, not evolution.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from self_assembling_games import model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("Use a clean committed checkout for the study")
    truth_tables = list(itertools.product((0, 1), repeat=4))
    protocol = {
        "study": "random_genome_boolean_repertoire",
        "revision": revision,
        "seed": args.seed,
        "genomes_per_gene_count": args.count,
        "genes": [2, 4],
        "response_updates": [2, 8],
        "panels": {
            "default": {},
            "rotated": {"input_0": "0110", "input_1": "1111", "act": "0001"},
        },
        "truth_tables": truth_tables,
        "independent_unit": "sampled genome; conditions reuse each genome",
        "selection": "none",
        "baseline": "best constant policy per truth table",
        "functional_threshold": 1e-12,
        "perfect_threshold": 1 - 1e-12,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    started = time.monotonic()
    for genes in protocol["genes"]:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, genes]))
        genomes = [
            "".join(map(str, row))
            for row in rng.integers(2, size=(args.count, 20 * genes))
        ]
        for updates in protocol["response_updates"]:
            for panel, ligands in protocol["panels"].items():
                declaration = model(
                    "toylife.boolean.nand",
                    realisation={
                        "parameters": {
                            "genes": genes,
                            "apparatus": {"ligands": ligands},
                        }
                    },
                    execution={
                        "parameters": {"response_updates": updates, "trace_limit": 0}
                    },
                )
                system = declaration.instantiate(args.seed)
                report = system.evaluate(genotypes=genomes)
                # case_metrics carries candidate-major action probability rows.
                case_rows = report["case_metrics"]
                probabilities = np.asarray(
                    [row["action_probabilities"] for row in case_rows]
                )[:, :, 1].T
                assert probabilities.shape == (args.count, 4)
                scores = np.asarray(
                    [
                        np.mean(
                            np.where(
                                np.asarray(table), probabilities, 1 - probabilities
                            ),
                            axis=1,
                        )
                        for table in truth_tables
                    ]
                ).T
                baseline = [max(sum(t), 4 - sum(t)) / 4 for t in truth_tables]
                record = {
                    "genes": genes,
                    "response_updates": updates,
                    "panel": panel,
                    "model": declaration.to_dict(),
                    "native_provenance": system.inspect()["provenance"],
                    "genomes": genomes,
                    "probability_one_by_input": probabilities.tolist(),
                    "scores_by_truth_table": scores.tolist(),
                    "work": report["work"],
                    "summary": {
                        "input_sensitive_genomes": int(
                            np.sum(np.ptp(probabilities, axis=1) > 1e-12)
                        ),
                        "best_score_by_truth_table": scores.max(axis=0).tolist(),
                        "mean_score_by_truth_table": scores.mean(axis=0).tolist(),
                        "best_constant_by_truth_table": baseline,
                        "above_constant_counts": np.sum(
                            scores > np.asarray(baseline) + 1e-12, axis=0
                        ).tolist(),
                        "perfect_counts": np.sum(scores >= 1 - 1e-12, axis=0).tolist(),
                    },
                }
                records.append(record)
                (args.output / "results.json").write_text(
                    json.dumps(
                        {
                            "protocol": protocol,
                            "conditions": records,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "genes": genes,
                            "updates": updates,
                            "panel": panel,
                            **record["summary"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
