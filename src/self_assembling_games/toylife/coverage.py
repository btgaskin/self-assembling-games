"""Case-level relationships between catalogue cases and chemical task assays.

Support describes the task interface, never empirical success or reproduction.
"""
from __future__ import annotations

import copy

# Each case owns its assumptions and outstanding requirements. In particular,
# a shared task name does not confer the source game's organisation or learner.
_CASES = [
    ("SIG", "derived_task", ["classification"], "Externally encoded classes and decoded acts replace sender and receiver conventions.", "Separate communicating participants and an explicit signal channel."),
    ("COD", "derived_task", ["classification"], "Four classes are presented to one chemical controller.", "Two independently produced messages and joint receiver interpretation."),
    ("NAND", "native_task", ["boolean"], "The NAND truth relation is externally scored; the genome supplies the response.", "Source sender and receiver organisation for a signalling reproduction."),
    ("HIER", "derived_task", ["context"], "Context is an input field rather than a learned activation hierarchy.", "Evolving activation, module interfaces and signed local feedback."),
    ("CUE", "derived_task", ["conditioning"], "A supplied cue predicts a supplied target.", "Fixed sender behaviour and independently adapted receiver."),
    ("MAN", "derived_task", ["classification"], "Classification with a fixed output permutation assays behavior under a supplied receiver mapping.", "An acting sender that changes a receiver's sensory input."),
    ("ORD", "derived_task", ["ordering"], "Numerical stimuli and relation scores are supplied.", "Signal invention and the source's first-failure row protocol."),
    ("TR-ORD", "derived_task", ["ordering"], "Ordering measures task accuracy with one inherited controller.", "Frozen old ordering template and separately adaptive input selectors."),
    ("TR-PART", "derived_task", ["ordering"], "Adjacent-pair training can probe withheld relations.", "Frozen template, adaptive selectors and recovered source pair sampler."),
    ("TR-LOG", "derived_task", ["boolean"], "Truth tables define fresh chemical tasks.", "Frozen NAND/OR templates with adaptive input translations and retention tests."),
    ("KIND", "analysis", ["classification"], "Classes and labels are supplied by the experiment.", "Derive descriptive partitions from confusion metrics; a conventional-kind analysis needs an operational criterion."),
    ("POLY", "needs_specification", [], "A genome is a single shared protein pool.", "A candidate connection grammar, composition dynamics and task feedback."),
    ("SCOD", "derived_task", ["context"], "Context selection is supplied in observations.", "Independent senders, learned source attention and receiver conventions."),
    ("COMP-NAND", "derived_task", ["boolean"], "A Boolean assay scores an end-to-end response.", "Frozen NAND modules and adaptive one- or two-selector wiring."),
    ("ADD", "derived_task", ["addition"], "Operand encoding and target sums are supplied.", "Invented signals, source operand sampling and source invention protocol."),
    ("ADD-COMP", "derived_task", ["sum_compare"], "The assay scores comparison after addition directly.", "Separately pretrained addition/comparison modules, translation and tie protocol."),
    ("ASK", "derived_task", ["query"], "A finite query interface supplies observation/reply timing.", "Source slab/block information sets and repeated-question costs."),
    ("DIAL", "derived_task", ["query"], "Query stages exercise one controller's conditional response.", "Two private observation holders, learned replies and turn taking."),
    ("DIAL-restricted", "derived_task", ["query"], "A query assay supplies a restricted response interface.", "Source two-answer restriction and independent dialogue participants."),
    ("DIAL-costly", "needs_specification", ["query"], "Chemical response windows have a finite configured horizon.", "Unbounded dialogue history, stopping, costs and negative-return protocol."),
    ("VIS", "needs_organisation", [], "Population selection evaluates individual genomes.", "Persistent agent identities, host choice and network payoffs."),
    ("STAG", "needs_organisation", [], "Chemical selection supplies no visiting graph.", "Reciprocal host visits, stag/hare payoffs and imitation specification."),
    ("FFSR", "needs_organisation", [], "Genome evolution acts on one controller type.", "Partner choice, complete signalling strategies, forgetting and exploration."),
    ("SUB", "needs_specification", [], "Input fields are supplied by an apparatus.", "Link-set actions, subscription costs and a specified adaptation rule."),
    ("DISC", "needs_organisation", [], "Task selection uses declared scalar scores.", "Bargaining strategies, persistent groups, partner choice and best responses."),
    ("INQ", "derived_task", ["advisers"], "Adviser responses are supplied task observations.", "Persistent epistemic network, self/nature consultation and truth-based local updates."),
    ("BOX", "native_task", ["boxes"], "Box rewards follow the configured constructed schedule.", "Declare fixed/random reward law when comparing with a source profile."),
    ("CHANGE", "derived_task", ["reversal"], "The target changes on a configured schedule.", "Source coding partitions and bounded reinforcement update."),
    ("INV", "needs_organisation", [], "The chemical input/output alphabet is apparatus-defined.", "Evolving signal alphabet, invention option and forgetting of types or tokens."),
    ("HYB", "needs_organisation", [], "Mutation and selection supply the adaptive mechanism.", "Within-encounter assignments and the WSLRwR failure update."),
    ("BAND", "native_task", ["bandit"], "Arm payoffs are explicitly configured.", "Supply the source machine reward distributions for a numerical comparison."),
    ("META-RULE", "needs_organisation", [], "One mutation/selection rule is configured per model.", "Selectable inner learning rules, fresh task boundaries and outer return."),
    ("META-ORACLE", "derived_task", ["advisers"], "Adviser quality and answers follow the configured assay.", "Specify source answer process, outer learning and reset boundaries."),
    ("MH", "native_task", ["monty_hall"], "The host and stay/switch interface follow the configured assay.", "Match source host protocol and distinguish chemical selection from Bayesian updating."),
    ("GEN", "derived_task", ["ordering", "classification"], "Withheld cases assess performance under a declared encoding.", "Ordering supplies an adjacent/nonadjacent split; specify a source similarity and reinforcement-spreading mechanism."),
    ("SET", "needs_specification", ["reversal"], "Reversal is a task sequence assay.", "Specify learning-set schedule, transferable state and self-tuning update."),
    ("ATT-R", "derived_task", ["context"], "All encoded fields reach one chemical controller.", "Receiver-controlled feature access and pinned distractor law."),
    ("ATT-B", "derived_task", ["context"], "Context fields provide a supplied observation interface.", "Both agents' attention choices, table sharing and distractor process."),
    ("RIT", "needs_specification", ["habituation"], "Repeated stimuli expose persistence and adaptation to selection.", "Primitive salience dynamics and a criterion for ritualized learning."),
    ("SK-SIG", "derived_task", ["classification"], "Classification supplies a dyadic task surrogate.", "Separate sender/receiver plans, state law and explicit cost interpretation."),
    ("SK-COMP", "derived_task", ["context"], "One controller can receive complementary input fields.", "Separate partial observers and independent signalling interfaces."),
    ("SK-VOTE", "derived_task", ["vote"], "A central controller aggregates externally corrupted observations.", "Independent sender policies, error law and signal-count receiver organisation."),
    ("SK-TEAM", "needs_organisation", [], "The current controller has one decoded response per decision.", "Addressed messages and coordinated actions by independent receivers."),
    ("SK-CHAIN", "needs_organisation", [], "Chemical updates share one protein pool.", "Relay participants, explicit channels and frozen endpoint translation."),
    ("SK-DIALOGUE", "derived_task", ["query"], "A supplied query protocol selects subsequent input.", "Resolve source dialogue information sets and define participant interfaces."),
    ("ML-COORD", "derived_task", ["classification"], "One genome supplies an end-to-end response.", "Two inherited code variants, population association and encounter matching."),
    ("ML-TRANSFER", "derived_task", ["boolean"], "NAND is an externally scored truth relation.", "Retained fixed NAND template plus two separately inherited input adapters."),
    ("ML-COMPOSE", "derived_task", ["boolean"], "Parity can be supplied as a Boolean truth table.", "Three preformed interface variants and their multilocus compatibility relation."),
    ("ECG", "needs_organisation", [], "Chemical task trials supply discrete observations and responses.", "Private colour boards, ordinary movement/finish actions, two controllers and acquired-signal measurements."),
]


def coverage() -> list[dict]:
    """Return a fresh JSON-compatible inventory of all 49 catalogue cases."""
    return copy.deepcopy([
        {"id": identifier, "support": support, "tasks": tasks,
         "assumptions": [assumption], "requirements": [requirement]}
        for identifier, support, tasks, assumption, requirement in _CASES
    ])
