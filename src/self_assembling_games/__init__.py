"""Barrett-style reinforcement games with inspectable, composable state."""

from .artifacts import export_template, replay_trace
from .core import Address, Decision, Feedback, Outcome, Runtime, Table
from .metrics import (
    conventional_kinds,
    evaluate,
    held_out_ordering,
    nand_transfer_closure,
)
from .registry import Profile, catalogue, load_profile, profiles, reproduction_protocols
from .simulation import (
    Simulation,
    benchmark,
    build,
    read_snapshot,
    resume,
    run_ensemble,
)

__version__ = "0.1.0"
__all__ = [
    "Address",
    "Decision",
    "Feedback",
    "Outcome",
    "Profile",
    "Runtime",
    "Simulation",
    "Table",
    "benchmark",
    "build",
    "catalogue",
    "conventional_kinds",
    "evaluate",
    "export_template",
    "held_out_ordering",
    "load_profile",
    "nand_transfer_closure",
    "profiles",
    "read_snapshot",
    "replay_trace",
    "reproduction_protocols",
    "resume",
    "run_ensemble",
]
