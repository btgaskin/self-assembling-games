"""The CLI resolves the same Model, Experiment and System objects as Python."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _legacy(argv):
    """Keep existing numerical-kernel command forms readable during migration."""
    command = argv[0] if argv else ""
    if command in {"list", "methods", "inspect", "ensemble", "benchmark", "replay"}:
        return True
    if command in {"run", "resume"} and any(
        flag in argv
        for flag in (
            "--plays",
            "--time",
            "--generations",
            "--trace",
            "--checkpoint-every",
        )
    ):
        return True
    if command == "run" and "--output" in argv:
        return True
    if command == "describe" and "--resolved" in argv:
        return True
    if command in {"resume", "evaluate"} and len(argv) > 1:
        path = Path(argv[1])
        return not (path / "manifest.json").is_file() and not (
            path.is_file() and path.suffix == ".toml"
        )
    return False


def _subject(value):
    from .config import load
    from .domain import model

    path = Path(value)
    return load(path) if path.is_file() else model(value)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if _legacy(argv):
        from ._legacy_cli import main as legacy_main

        return legacy_main(argv)
    parser = argparse.ArgumentParser(
        prog="sg", description="Resolve, run and archive self-assembling games"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "catalogue", help="List source-grounded built-in and situated examples"
    )
    for name in ("describe", "resolve"):
        sub.add_parser(name).add_argument(
            "subject", help="Model identifier or TOML path"
        )
    init = sub.add_parser(
        "init", help="Create workspace directories and an editable experiment TOML"
    )
    init.add_argument("directory", nargs="?", default=".")
    init.add_argument("--model", default="sig.demo")
    init.add_argument("--budget", type=int, default=100)
    running = sub.add_parser(
        "run", help="Run a model or TOML study in a new timestamped directory"
    )
    running.add_argument("subject")
    running.add_argument("--directory", default="runs")
    running.add_argument("--workspace")
    running.add_argument("--budget", type=float)
    running.add_argument("--seeds", type=int, nargs="+")
    running.add_argument("--workers", type=int)
    for name in ("status", "verify", "evaluate"):
        sub.add_parser(name).add_argument("directory")
    continuing = sub.add_parser("resume")
    continuing.add_argument("directory")
    continuing.add_argument("--additional-budget", type=float)
    args = parser.parse_args(argv)
    try:
        from .config import save
        from .domain import Experiment, experiment
        from .experiments import Run, Workspace, resume, run

        if args.command == "catalogue":
            from .registry import catalogue, profiles
            from .finite_profiles import adaptive_profiles, finite_catalogue
            from .situated import ECG_MODEL, situated_catalogue
            from .toylife import (
                presets as molecular_presets,
                coverage as molecular_coverage,
            )

            result = {
                "games": catalogue() + finite_catalogue() + situated_catalogue(),
                "presets": profiles()
                + adaptive_profiles()
                + [{"id": ECG_MODEL["id"], "runtime": "situated"}]
                + molecular_presets(),
                "molecular_coverage": molecular_coverage(),
            }
        elif args.command in {"describe", "resolve"}:
            result = _subject(args.subject).to_dict()
        elif args.command == "init":
            destination = Path(args.directory)
            target = destination / "experiment.toml"
            if target.exists():
                raise ValueError(f"Configuration already exists: {target}")
            Workspace(destination).init()
            save(
                {
                    "schema_version": 2,
                    "kind": "experiment",
                    "preset": args.model,
                    "seeds": [0],
                    "budget": args.budget,
                },
                target,
            )
            result = {"workspace": str(destination), "configuration": str(target)}
        elif args.command == "run":
            subject = _subject(args.subject)
            if not isinstance(subject, Experiment):
                subject = experiment(subject)
            if (
                args.budget is not None
                or args.seeds is not None
                or args.workers is not None
            ):
                data = subject.to_dict()
                if args.budget is not None:
                    for condition in data["conditions"]:
                        condition["budget"] = (
                            int(args.budget)
                            if args.budget.is_integer()
                            else args.budget
                        )
                if args.seeds is not None:
                    data["seeds"] = args.seeds
                if args.workers is not None:
                    data["workers"] = args.workers
                subject = Experiment.from_dict(data)
            elif Path(args.subject).is_file():
                subject = args.subject
            original = (
                Path(args.subject).read_text(encoding="utf-8")
                if Path(args.subject).is_file()
                else None
            )
            study = run(
                subject,
                directory=args.directory,
                workspace=args.workspace,
                original=original,
            )
            result = {"path": str(study.path), **study.status()}
        elif args.command == "resume":
            amount = args.additional_budget
            if amount is not None and amount.is_integer():
                amount = int(amount)
            study = resume(args.directory, additional_budget=amount)
            result = {"path": str(study.path), **study.status()}
        elif args.command == "status":
            result = Run.load(args.directory).status()
        elif args.command == "verify":
            result = Run.load(args.directory).verify()
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["valid"] else 1
        else:
            result = Run.load(args.directory).evaluate()
    except (ValueError, KeyError, OSError, TypeError, ImportError) as error:
        print(f"sg: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("status") == "complete_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
