"""A game, its organisation, its realised dispositions and its specification."""

from __future__ import annotations

import copy


class ModelInstance:
    """The objects which execute a constructed model.

    A specification records initial choices. The realisation holds the current
    state, including changes made by interventions and experience.
    """

    def __init__(self, specification, game, organisation, realisation):
        self.specification = specification
        self._game = game
        self.organisation = organisation
        self.realisation = realisation
        self._parameters = specification.game_parameters
        if "owners" in specification.organisation_parameters:
            self._parameters["owners"] = specification.organisation_parameters["owners"]
        self._learner = specification.realisation_parameters

    @property
    def game(self):
        return (
            self._game
            if self.realisation.id == "barrett.conditional"
            else self.organisation._game
        )

    @property
    def model(self):
        return self.specification.game_id

    @property
    def parameters(self):
        return copy.deepcopy(self._parameters)

    @property
    def learner(self):
        return copy.deepcopy(self._learner)

    @property
    def runtime(self):
        return self.realisation

    @property
    def world(self):
        return self.realisation.world

    @property
    def interaction(self):
        return self.realisation.interaction

    @property
    def lineage(self):
        return self.realisation.lineage

    def initialize(self):
        if self.realisation.id != "barrett.conditional":
            raise ValueError("Finite realised state is initialised during construction")
        self.game.initialize(self)

    def encounter(self):
        if self.realisation.id == "barrett.conditional":
            self.realisation.before_encounter(self._learner)
            return self.game.encounter(self)
        return self.realisation.encounter(self.game)

    def inspect(self):
        return {
            "specification": self.specification.to_dict(),
            "organisation": self.organisation.to_dict(),
            "state_addresses": self.organisation.inventory(self.realisation),
            "decision_bindings": self.organisation.decision_bindings(self.realisation),
            "realisation": self.realisation.inspect_dispositions(),
            "construction_provenance": copy.deepcopy(
                getattr(self.realisation, "lineage", {})
            ),
        }
