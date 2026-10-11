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
        prog="sg",
        description="Self-assembling games: learning and evolutionary dynamics",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--experiments", action="store_true")
    listing.add_argument(
        "--family", choices=["barrett", "skyrms", "constructed", "all"], default="all"
    )
    sub.add_parser(
        "methods", help="List adaptive method classes and their named methods"
    )
    describe = sub.add_parser("describe")
    describe.add_argument("identifier")
    describe.add_argument(
        "--resolved",
        action="store_true",
        help="Construct and inspect resolved choices and initial realised state",
    )
    describe.add_argument("--seed", type=int, default=0)
    running = sub.add_parser("run")
    running.add_argument("profile")
    running.add_argument("--seed", type=int, default=0)
    running.add_argument("--plays", type=int)
    running.add_argument("--time", type=float, help="Additional evolutionary time")
    running.add_argument("--generations", type=int)
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
    continuing.add_argument("--plays", type=int)
    continuing.add_argument("--time", type=float)
    continuing.add_argument("--generations", type=int)
    continuing.add_argument("--output", type=Path)
    ensemble = sub.add_parser("ensemble")
    ensemble.add_argument("profile")
    ensemble.add_argument("--seeds", type=int, nargs="+", required=True)
    ensemble.add_argument("--plays", type=int)
    ensemble.add_argument("--time", type=float)
    ensemble.add_argument("--generations", type=int)
    ensemble.add_argument("--metric")
    ensemble.add_argument("--threshold", type=float)
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("--workers", type=int, default=1)
    timing = sub.add_parser("benchmark")
    timing.add_argument("profile")
    timing.add_argument("--plays", type=int)
    timing.add_argument("--time", type=float)
    timing.add_argument("--generations", type=int)
    replay = sub.add_parser("replay")
    replay.add_argument("trace")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            from .finite_profiles import adaptive_profiles, finite_catalogue

            if args.experiments:
                from .registry import reproduction_protocols

                if args.family == "skyrms":
                    raise ValueError(
                        "Historical adaptive experiments remain source-gated; SKQ1"
                    )
                result = reproduction_protocols()
            elif args.family in {"skyrms", "constructed"}:
                result = (
                    finite_catalogue(args.family)
                    if args.all
                    else adaptive_profiles(args.family)
                )
            elif args.family == "all":
                result = (
                    catalogue()
                    + finite_catalogue("skyrms")
                    + finite_catalogue("constructed")
                    if args.all
                    else profiles() + adaptive_profiles()
                )
            else:
                result = catalogue() if args.all else profiles()
        elif args.command == "methods":
            from .adaptive import adaptive_methods

            result = adaptive_methods()
        elif args.command == "describe":
            if args.resolved:
                print(
                    json.dumps(
                        build(args.identifier, args.seed).model_instance.inspect(),
                        indent=2,
                        allow_nan=False,
                    )
                )
                return 0
            entries = [x for x in catalogue() if x["id"] == args.identifier]
            from .finite_profiles import (
                adaptive_profiles,
                finite_catalogue,
                load_adaptive_profile,
            )

            finite_entries = [
                x for x in finite_catalogue() if x["id"] == args.identifier
            ]
            if entries:
                result = entries[0]
            elif finite_entries:
                result = finite_entries[0]
            elif any(p["id"] == args.identifier for p in adaptive_profiles()):
                result = load_adaptive_profile(args.identifier).to_dict()
            else:
                from .adaptive_runtime import is_adaptive_profile

                result = (
                    load_adaptive_profile(args.identifier).to_dict()
                    if is_adaptive_profile(args.identifier)
                    else load_profile(args.identifier).to_dict()
                )
        elif args.command == "run":
            if (args.output / "checkpoint.json").exists():
                raise ValueError(
                    "Output already has a checkpoint; use resume or choose a fresh directory"
                )
            sim = build(args.profile, args.seed)
            result = _run(
                sim,
                plays=args.plays,
                duration=args.time,
                generations=args.generations,
                output=args.output,
                checkpoint_every=args.checkpoint_every,
                trace=args.trace,
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
            if args.plays is None and args.time is None and args.generations is None:
                raise ValueError("Resume requires --plays, --time, or --generations")
            result = _run(
                sim,
                plays=args.plays,
                duration=args.time,
                generations=args.generations,
                output=args.output or args.checkpoint,
            )
        elif args.command == "ensemble":
            result = run_ensemble(
                args.profile,
                args.seeds,
                output=args.output,
                plays=args.plays,
                threshold=args.threshold,
                workers=args.workers,
                duration=args.time,
                generations=args.generations,
                metric=args.metric,
            )
        elif args.command == "replay":
            from .artifacts import replay_trace

            result = replay_trace(args.trace).metrics.summary()
        else:
            result = benchmark(
                args.profile,
                args.plays,
                duration=args.time,
                generations=args.generations,
            )
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"sg: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _run(
    sim,
    *,
    plays=None,
    duration=None,
    generations=None,
    output=None,
    checkpoint_every=0,
    trace=None,
):
    from .evolution import EvolutionSimulation
    from .finite_learning import ReinforcementSimulation
    from .multilocus import MultilocusSimulation

    if isinstance(sim, MultilocusSimulation):
        if plays is not None or duration is not None or checkpoint_every or trace:
            raise ValueError(
                "Multilocus populations use --generations and census checkpoints"
            )
        return sim.run(generations=generations, output=output)
    if generations is not None:
        raise ValueError("--generations belongs to multilocus generation methods")

    if isinstance(sim, EvolutionSimulation):
        if plays is not None or checkpoint_every or trace:
            raise ValueError(
                "Evolution uses --time; adaptive event traces/checkpoint-every are unavailable"
            )
        return sim.run(duration=duration, output=output)
    if duration is not None:
        raise ValueError(
            "Reinforcement uses --plays; --time belongs to population evolution"
        )
    if isinstance(sim, ReinforcementSimulation):
        if checkpoint_every or trace:
            raise ValueError(
                "Adaptive reinforcement event traces/checkpoint-every are unavailable"
            )
        return sim.run(plays=plays, output=output)
    return sim.run(
        plays=plays, output=output, checkpoint_every=checkpoint_every, trace=trace
    )


if __name__ == "__main__":
    raise SystemExit(main())
