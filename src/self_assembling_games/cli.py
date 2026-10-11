"""The CLI uses the same profiles, lifecycle, and measurements as Python."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import evaluate, held_out_ordering
from .registry import catalogue, load_profile, profiles
from .simulation import benchmark, build, read_snapshot, resume, run_ensemble


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sg", description="Source-based self-assembling games"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--experiments", action="store_true")
    describe = sub.add_parser("describe")
    describe.add_argument("identifier")
    running = sub.add_parser("run")
    running.add_argument("profile")
    running.add_argument("--seed", type=int, default=0)
    running.add_argument("--plays", type=int)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--trace", type=Path)
    running.add_argument("--checkpoint-every", type=int, default=0)
    inspecting = sub.add_parser("inspect")
    inspecting.add_argument("checkpoint")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument(
        "--protocol", choices=["exact", "sampled", "held-out"], default="exact"
    )
    evaluation.add_argument("--samples", type=int, default=1000)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-branches", type=int, default=200000)
    continuing = sub.add_parser("resume")
    continuing.add_argument("checkpoint")
    continuing.add_argument("--plays", type=int, required=True)
    continuing.add_argument("--output", type=Path)
    ensemble = sub.add_parser("ensemble")
    ensemble.add_argument("profile")
    ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    ensemble.add_argument("--plays", type=int)
    ensemble.add_argument("--threshold", type=float)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--workers", type=int, default=1)
    timing = sub.add_parser("benchmark")
    timing.add_argument("profile")
    timing.add_argument("--plays", type=int, default=10000)
    replay = sub.add_parser("replay")
    replay.add_argument("trace")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            if args.experiments:
                from .registry import reproduction_protocols

                result = reproduction_protocols()
            else:
                result = catalogue() if args.all else profiles()
        elif args.command == "describe":
            entries = [x for x in catalogue() if x["id"] == args.identifier]
            result = entries[0] if entries else load_profile(args.identifier).to_dict()
        elif args.command == "run":
            if (args.output / "checkpoint.json").exists():
                raise ValueError(
                    "Output already has a checkpoint; use resume or choose a fresh directory"
                )
            sim = build(args.profile, args.seed)
            result = sim.run(
                plays=args.plays,
                output=args.output,
                trace=args.trace,
                checkpoint_every=args.checkpoint_every,
            )
        elif args.command == "inspect":
            result = resume(args.checkpoint).inspect()
        elif args.command == "evaluate":
            snapshot = read_snapshot(args.checkpoint)
            result = (
                held_out_ordering(snapshot, samples=args.samples, seed=args.seed)
                if args.protocol == "held-out"
                else evaluate(
                    snapshot,
                    args.protocol,
                    samples=args.samples,
                    seed=args.seed,
                    max_branches=args.max_branches,
                )
            )
        elif args.command == "resume":
            sim = resume(args.checkpoint)
            result = sim.run(plays=args.plays, output=args.output or args.checkpoint)
        elif args.command == "ensemble":
            result = run_ensemble(
                args.profile,
                args.seeds,
                output=args.output,
                plays=args.plays,
                threshold=args.threshold,
                workers=args.workers,
            )
        elif args.command == "replay":
            from .artifacts import replay_trace

            result = replay_trace(args.trace).metrics.summary()
        else:
            result = benchmark(args.profile, args.plays)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"sg: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
