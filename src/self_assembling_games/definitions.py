"""Registered executable definitions; procedural kernels supply each encounter."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from importlib import import_module


_FAMILIES = {
    "signalling": ("SIG", "CUE", "MAN", "HYB"),
    "coding": ("COD", "SCOD", "CHANGE"),
    "nand": ("NAND",),
    "hierarchy": ("HIER",),
    "invention": ("INV", "ORD", "ADD"),
    "composition": ("TR-LOG", "COMP-NAND"),
    "transfer": ("TR-ORD", "TR-PART", "ADD-COMP"),
    "dialogue": ("DIAL", "DIAL-restricted", "DIAL-costly", "ASK"),
    "network": ("VIS", "STAG", "FFSR", "INQ"),
    "box": ("BOX",),
    "task": ("BAND", "META-ORACLE", "META-RULE", "MH"),
    "attention": ("ATT-R", "ATT-B"),
    "generalization": ("GEN-APPROX",),
}


@dataclass(frozen=True)
class GameDefinition:
    id: str
    family: str
    implementation: str

    def validate(self, profile):
        if self.implementation == "finite.factory":
            profile.validate()
        else:
            from .validation import validate_barrett

            validate_barrett(profile)
            self.kernel.validate(profile)

    @property
    def kernel(self):
        return import_module(".game_families." + self.family, __package__)

    def create_finite(self, profile):
        from .finite_profiles import finite_game

        self.validate(profile)
        return finite_game(profile)

    def resolve_parameters(self, parameters):
        parameters = copy.deepcopy(parameters)
        resolver = getattr(self.kernel, "resolve_parameters", None)
        return resolver(self.id, parameters) if resolver else parameters

    def initial_organisation(self, parameters):
        from .registry import catalogue

        entry = next(m for m in catalogue() if m["id"] == self.id)
        owners, roles = {}, {}
        includes = getattr(
            self.kernel, "includes_module", lambda model, module, parameters: True
        )
        for module, descriptor in entry.get("modules", {}).items():
            if not includes(self.id, module, parameters):
                continue
            owners[module] = parameters.get("owners", {}).get(
                module, descriptor["owner"]
            )
            roles[module] = descriptor["role"]
        return {"owners": owners, "functional_roles": roles}

    def initialize(self, model):
        declaration = model.specification.organisation_parameters
        for module, owner in declaration["owners"].items():
            model.organisation.declare(
                module, owner, declaration["functional_roles"][module]
            )
        self.kernel.initialize(model)

    def encounter(self, model):
        return self.kernel.encounter(model)


_BARRETT = {
    identifier: GameDefinition(identifier, family, "barrett." + family)
    for family, identifiers in _FAMILIES.items()
    for identifier in identifiers
}


def game_definition(identifier, realisation="barrett.conditional") -> GameDefinition:
    if isinstance(identifier, GameDefinition):
        registered = game_definition(identifier.id, realisation)
        if identifier != registered:
            raise ValueError(
                "Game definition differs from its registered implementation"
            )
        return registered
    if realisation == "barrett.conditional":
        if identifier not in _BARRETT:
            from .registry import catalogue

            entries = {m["id"]: m for m in catalogue()}
            if identifier in entries:
                raise ValueError(
                    f"{identifier} is a catalogue entry awaiting an operational protocol: "
                    + ", ".join(entries[identifier]["required_fields"])
                )
            raise ValueError(f"Unknown model {identifier}")
        return _BARRETT[identifier]
    from .finite_profiles import finite_catalogue

    entry = next((g for g in finite_catalogue() if g["id"] == identifier), None)
    if entry is None:
        raise ValueError(f"Unknown finite game {identifier}")
    return GameDefinition(identifier, entry["family"], "finite.factory")
