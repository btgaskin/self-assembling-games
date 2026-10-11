"""Self-assembling games with inspectable learning and evolutionary dynamics."""

from .artifacts import export_template, replay_trace
from .core import Address, Decision, Feedback, Outcome, Runtime, Table
from .definitions import GameDefinition, game_definition
from .specification import (
    ModelSpecification,
    RealisationDefinition,
    SourceDeclaration,
    realisation_definition,
)
from .model import ModelInstance
from .organisation import Organisation
from .organisation import MultilocusPopulationSpec
from .genetics import Locus, GenomeSpec, ExpressionMap, TransmissionSpec
from .multilocus import MultilocusRealisation, MultilocusSimulation
from .realisations import FiniteReinforcementRealisation, PopulationRealisation
from .metrics import (
    conventional_kinds,
    evaluate,
    held_out_ordering,
    nand_transfer_closure,
)
from .registry import Profile, catalogue, load_profile, profiles, reproduction_protocols
from .adaptive import AdaptiveMethod, adaptive_method, adaptive_methods
from .adaptive_runtime import build_adaptive, restore_adaptive
from .finite import FiniteDecision, FiniteGame, ParticipantOutcome
from .finite_adapters import barrett_game
from .finite_profiles import (
    AdaptiveProfile,
    adaptive_profiles,
    finite_catalogue,
    finite_game,
    load_adaptive_profile,
    register_finite_game,
)
from .evolution import EvolutionSimulation, PopulationSpec
from .finite_learning import ReinforcementSimulation
from .finite_artifacts import import_frozen_decisions
from .adaptive_experiments import adaptive_ensemble
from .skyrms import skyrms_game
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
    "Locus",
    "GenomeSpec",
    "ExpressionMap",
    "TransmissionSpec",
    "MultilocusPopulationSpec",
    "MultilocusRealisation",
    "MultilocusSimulation",
    "GameDefinition",
    "game_definition",
    "ModelSpecification",
    "RealisationDefinition",
    "SourceDeclaration",
    "realisation_definition",
    "ModelInstance",
    "Organisation",
    "FiniteReinforcementRealisation",
    "PopulationRealisation",
    "AdaptiveMethod",
    "AdaptiveProfile",
    "EvolutionSimulation",
    "FiniteDecision",
    "FiniteGame",
    "ParticipantOutcome",
    "PopulationSpec",
    "ReinforcementSimulation",
    "adaptive_method",
    "adaptive_methods",
    "adaptive_profiles",
    "barrett_game",
    "build_adaptive",
    "finite_catalogue",
    "finite_game",
    "load_adaptive_profile",
    "register_finite_game",
    "restore_adaptive",
    "skyrms_game",
    "adaptive_ensemble",
    "import_frozen_decisions",
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
