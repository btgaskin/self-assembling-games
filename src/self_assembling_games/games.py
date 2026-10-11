"""Compatibility entry points for the registered game mechanisms."""

from .game_families.common import (
    expected_signaling,
    forget_signal,
    ordering_call,
    relation,
)


__all__ = [
    "expected_signaling",
    "forget_signal",
    "ordering_call",
    "relation",
    "initialize",
    "play",
]


def initialize(sim):
    sim.model_instance.initialize()


def play(sim):
    return sim.model_instance.encounter()
