# Games, organisation and realised dispositions

Every executable simulation has a `model_instance`. Its four objects have
specific responsibilities in construction and execution:

| Object | Responsibility | Live examples |
|---|---|---|
| `game` | Conditions, permitted observations, choice sequence and feedback assignment | A received expression; a translator invoking a frozen NAND template; an asynchronous inquiry consultation |
| `organisation` | Component ownership, functional roles, encounter bindings and population grouping | Two modules owned by one participant; a distinct individual in each encounter slot; jointly inherited role plans |
| `realisation` | State and operations sustaining conditional and adaptive dispositions | Propensities, assignments, reference points, beliefs, dynamic options and inherited-plan frequencies |
| `specification` | Resolved initial choices and their source declarations | State distributions, bindings, learner settings, initialisation procedures and named source questions |

The game controls construction order. The realisation creates its adaptive state,
and organisation resolves component relations. During an encounter the procedural
game kernel supplies observations and feedback addresses. The realisation selects
acts and applies the relevant adaptive operation. The simulation maintains run
controls, measurements, history and checkpoints.

## Constructing a model

```python
from self_assembling_games import ModelSpecification, SourceDeclaration, build

specification = ModelSpecification.resolve(
    game="SIG",
    game_parameters={"states": 2, "signals": 2, "actions": 2},
    organisation={"owners": {"sender": "animal", "receiver": "animal"}},
    realisation="barrett.conditional",
    realisation_parameters={"rule": "R1"},
    source=SourceDeclaration(
        kind="deliberate_variant",
        assumptions=("Sender and receiver are functional components of one animal.",),
    ),
)
simulation = build(specification, seed=7, protocol={"plays": 1000, "window": 100})
model = simulation.model_instance
print(model.organisation.owners)
print(model.realisation.tables["sender/0"].conditional_disposition())
simulation.run()
```

`ModelSpecification.from_profile("sig.n2.demo")` compiles an established profile
into this structure. Profile and native construction share scientific validators:
the source gates, supported learning rules, fixed components, resource budgets
and finite-game compatibility conditions apply to both paths. Registered custom
finite factories use the same resolution route.

The specification resolves effective game and learner defaults. Its accessors
return independent nested values. Initial distributions and procedures are
seed-independent; a constructed realisation contains the actual seeded
assignments and imported state. Existing profile JSON remains available for
configuration hashes and compatibility.

`build(..., protocol=...)` supplies outer run controls. Initial propensity mass,
population initialisation and mutation rate belong to the realisation. R3 phase
length, inquiry problem duration, CHANGE's transition time and META-RULE's
`inner_plays` remain model conditions: each determines what happens or what
feedback is produced. Solvers, reporting cadence, outer duration and measurement
window govern the experiment. Checkpoints serialize both accounts.

## Conditional and adaptive dispositions

A **conditional disposition to act** describes the current choice at a permitted
context. An **adaptive disposition** concerns how experience changes those
choices. Their realisation includes the mechanisms and state supporting both.

For R1, weights `(1, 1)` and `(100, 100)` both yield probabilities `(0.5, 0.5)`.
Adding one to the first option changes those probabilities to `(2/3, 1/3)` and
`(101/201, 100/201)`. Present choice probabilities therefore leave out a feature
of the adaptive disposition: retained propensity magnitude.

```python
table = model.realisation.tables["sender/0"]
table.conditional_disposition()   # acts under current assignments and error law
table.propensity_distribution()  # normalised retained mass
table.inspect_dispositions()    # both quantities, adaptive rule and scope
```

R6's failed choice draws a new assignment from the propensity distribution.
WSLS switches between its two options; WSLR draws uniformly. Whole-strategy
WSLR changes all unfrozen assignments in order. Current assignments and the
adaptive operation make these distinctions inspectable. The compatibility
spelling `probabilities(response=False)` returns the propensity distribution.

Five registered realisations cover the current executable cases:

| Identifier | State and change |
|---|---|
| `barrett.conditional` | Context-bound tables, assignments, reference points and world state; source-specific updates, invention and boundary operations |
| `finite.conditional` | Context-bound act propensities; validate an encounter's entire credit batch, then apply R1 |
| `finite.complete_plan` | Mass over complete plans; sample one plan per participant and reinforce it once |
| `population.complete_plan` | Joint inherited-plan frequencies; selection and the declared birth-mutation law under explicit matching |
| `population.multilocus` | Haplotype frequencies or counts; genotype-to-plan expression, individual participation, selection and explicit transmission |

Barrett feedback updates addressed tables sequentially. Finite reinforcement
validates and applies a whole credit transaction. Each implementation retains its
existing failure behavior and event order. Complete-plan selection distributions
and population encounter-plan distributions have their own inspection names;
their joint state retains associations between conditional acts.

## Ownership, functional roles and encounter slots

A module owner identifies the participant to which a component belongs. A
functional role describes its contribution, such as sending or translating.
An encounter slot specifies where a participant's policy is sampled. The legacy
`FiniteDecision.role` field names that slot; `FiniteDecision.slot` provides the
explicit spelling.

Organisation resolves finite decisions and credit addresses through the operative
game records. Its slot view is derived from those records. For populations,
`PopulationSpec` owns grouping, role weights, encounter rates and matching.
Several slots assigned to one population still draw distinct independent
individuals. Multiple modules inside one slot can share a complete plan, retaining
their internal association. Joint inheritance and joint participation are thus
separately specified.

For multilocus models, `MultilocusPopulationSpec` instead binds several encounter
slots to one sampled individual. All those slots express that individual's
genotype. Separate groups sample with replacement from the single haploid
population. [Multilocus methods](multilocus.md) specifies the expression,
conditional fitness and transmission equations, with their source references.

`organisation.inventory(realisation)` discovers current addresses directly from
live state. Invention, rollback and removal immediately alter that inventory.
`runtime.tables`, `runtime.owners`, `runtime.module_roles`, finite `weights` and
population `frequencies` expose the same backing objects used by the model.
Replacing a Barrett `runtime` rebinds its model. An adaptive game replacement is
constructed through `game=`, which validates and records the new game semantics.

## Inspecting and continuing

```sh
sg describe sig.n2.demo --resolved --seed 7
sg describe sk-sig.shared.demo --resolved --seed 7
```

Resolved description constructs initial state and advances no encounters. It
reports the scientific specification, operative organisation, current
dispositions, state addresses, decision bindings and construction provenance.
Researcher inspection leaves agents' permitted observations intact.

`snapshot()` records continuation state, including RNG, assignments, beliefs,
interventions, measurements and phase history. A specification records the initial
model. Template construction provenance retains artifact hashes and export
transforms. Barrett and finite-adapter checkpoints retain embedded template state
so restore can continue after the source artifact has been moved. Custom finite
factories retain their established registration requirement; explicit game
overrides can be supplied again when restoring.

## Research use and verification

The [executable example](../examples/realised_dispositions.py) compares equal
present dispositions with different R1 adaptation, then compares correlated and
anticorrelated complete plans. The same architecture supports a fixed-template
transfer with explicit adaptive adapters and retained module state.

These are computational realisations of the specified games. Complete-plan
population methods use continuous selection and uniform birth mutation.
Multilocus methods add an explicit inherited encoding, deterministic expression,
recombination and locus mutation under discrete generations. A molecular
genotype-to-organisation map requires its own concrete mechanisms. A biological
interpretation specifies the reproducing entities and how they make and modify
their dispositions.

The refactor is checked against pre-change state, RNG, events and trajectories for
the 85 executable profiles present before that refactor. Additional tests compare native and profile
construction, live aliases, source gates, failed credit transactions, template
restore, inspection and evaluation. Continuous solver arithmetic and tolerances
retain their established contract. Baseline hashes and their capture conditions
are stored in [tests/data](../tests/data/README.md).
