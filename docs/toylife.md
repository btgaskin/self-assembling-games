# toyLIFE chemical task assays

The adapter places a toyLIFE genome inside a declared observation and response
apparatus. Each genome supplies a **single shared protein pool**. Genes contribute
to that pool; their number is a capacity parameter. Molecular roles such as cue
retention, conditional control or coupled transformation can emerge through
chemical interactions and be inferred from behavior and interventions afterward.
Independent participants, compartments and selection units are separate
organisational choices. The present adapter evolves one organism with one pool.

The canonical discrete toyLIFE rules govern protein expression, binding and
chemical updates. A response window advances those rules and decodes a chemical
readout into an action. Continuous chemistry would be a separate future model
with its own equations and numerical protocol.

## Installation and model configuration

Use Python 3.12 or 3.13 for toyLIFE2. The base library supports Python 3.11;
the `toylife` extra requires the dependency's narrower Python range. Install
`python -m pip install -e '.[toylife]'` when toyLIFE2 is available to your package
index. For the local toyLIFE2 installation, install both projects
into the same environment; an editable installation keeps the local chemistry
implementation available without copying it into this repository:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e /path/to/toyLIFE2
.venv/bin/python -m pip install -e '.[dev]'
```

Presets have the form `toylife.<task>`. The Boolean presets
`toylife.boolean.nand`, `toylife.boolean.xor`, `toylife.boolean.and` and
`toylife.boolean.or` specify common truth tables. All sixteen binary functions
can be supplied as `truth_table = [a, b, c, d]`, ordered by inputs
`00`, `01`, `10`, `11`.

A resolved model declares `interaction.game` and task settings directly in
`interaction.parameters`. It declares `genes`, `apparatus` and `controller` in
`realisation.parameters`; `population_size` and `mutation_probability` in
`adaptation.parameters`; and `backend` and `response_updates` in
`execution.parameters`. Small presets use two genes, four candidate genomes,
mutation probability 0.01, eight response updates and the CPU backend.

Optional `initial_genotypes` in `realisation.parameters` supplies equal-length
binary strings of `20 * genes` bits: supply one founder string to replicate
across the initial population, or one string per initial organism. Two genes
therefore require 40 bits per string. These strings encode inherited genes;
`controller` declares the initial chemical state separately.

`fitness_floor` in `adaptation.parameters` defaults to `1e-8` and must lie in
`[0, 1)`. Selection fitness is `max(task_score, fitness_floor)`. The evaluator
rounds this to integer weights with denominator `2**24`; a positive floor gives
every candidate at least one weight unit. Raw task scores remain available for
performance analysis.

Input encoding is **per field**. Each declared observation field has its own
molecular channels and a `one_hot`, Boolean `presence`, or integer `dose`
encoding. Dose bounds and observation alphabets are part of the apparatus.
Omitted fields can represent absence during a delay. The apparatus allocates
channels from declared domains; it does not encode a complete observation tuple
as a memorized case label. Output markers, representation, decoding rule and
optional action permutation specify what chemical readouts mean.

Within an episode, the chemistry persists across observation, delay, action and
feedback phases. This allows experience to change the chemical state while the
genome stays fixed. The default resets chemistry at episode birth. Setting
`realisation.parameters.reset_between_phases = true` creates a memory control:
it resets the chemical state before every phase, including feedback, rest and
probe phases, while world state and random streams continue. The experience
workspace compares this control with retained chemistry at the same eight-update
delay. Task scores guide population selection between evaluations.
An observed reward can also arrive as a ligand in an explicit feedback phase;
that physical input can alter later chemistry. Selection scores and feedback
ligands therefore have different causal roles. Conditioning and habituation are
scheduled response assays whose changes must be measured from the resulting
trajectories.

## Tasks and smoke workspaces

The task interfaces are `classification`, `boolean`, `context`, `vote`,
`ordering`, `addition`, `sum_compare`, `monty_hall`, `delay`, `boxes`, `bandit`,
`advisers`, `query`, `conditioning`, `habituation` and `reversal`.
Classification, truth tables, contextual choice, voting, ordering and arithmetic
supply finite distributions with explicit scores. Episodic assays supply their
own hidden state and observation/feedback schedule. The query assay exposes two
binary attributes through a bounded interface that retains the latest answer. Each is a constructed
chemical-controller experiment with its own interface assumptions.

Four experiment workspaces use seed 0, two generations of budget and one worker:

| Workspace | Conditions | Purpose |
|---|---|---|
| `runs/toylife-capacity` | two/four genes | Change inherited chemical capacity on NAND |
| `runs/toylife-experience` | delay zero/eight; eight with phase resets | Probe the contribution of retained chemical state |
| `runs/toylife-apparatus` | declared/permuted binary output | Check dependence on supplied response meaning |
| `runs/toylife-transfer` | all/adjacent ordering pairs | Probe withheld ordering relations |

The transfer workspace compares ordering trained on all pairs with ordering
trained on adjacent pairs, using `train_pairs = "all"` or `"adjacent"`.
Its analysis requests `split = "held_out"`: nonadjacent unequal pairs provide
the withheld relation set for ordering. The `toylife.reversal` preset supplies a separate
within-episode target-change assay. Frozen-template transfer requires a retained
template and separately adaptive input mappings. These smoke budgets check execution and produce
reviewable outputs. Performance estimates require a declared evaluation protocol
and independent seeds.

Overrides in experiment TOML use
`[conditions.overrides.interaction.parameters]`,
`[conditions.overrides.realisation.parameters]`,
`[conditions.overrides.execution.parameters]` and
`[conditions.overrides.adaptation.parameters]`. The apparatus workspace uses
`[conditions.overrides.realisation.parameters.apparatus.output]` with
`permutation = [1, 0]`.

## Custom staged interactions

A custom `Interaction` declares scalar observation fields, actions and actor
`cell`. The example below, saved as importable `my_assay.py`, exposes a cue for
three chemical updates and then reads a response after cue withdrawal:

```python
from self_assembling_games.situated import Stage, StageResult

class CueProbe:
    actors = ("cell",)
    fields = {"cue": [0, 1]}
    actions = ["quiet", "respond"]
    def initial_world(self, rng): return {"phase": 0}
    def stage(self, world):
        return Stage("probe", () if world["phase"] == 0 else self.actors,
                     duration=3 if world["phase"] == 0 else 0)
    def observations(self, world):
        return {"cell": {"cue": 1} if world["phase"] == 0 else {}}
    def legal_actions(self, world, actor): return self.actions
    def transition(self, world, actions, rng):
        done = world["phase"] == 1
        return StageResult({"phase": 1}, terminated=done,
            feedback={"cell": {"score": float(actions.get("cell") == "respond")}} if done else {})

def factory(parameters): return CueProbe()
```

Replace a resolved model's `interaction` with
`{"provider": "my_assay:factory", "parameters": {}}` and resolve it with
`config.from_dict`. Omit `game` for this custom provider. `duration=None` uses
`response_updates`; an integer duration requests that many chemical updates,
including zero for an immediate readout. Keep retained world state in the returned
JSON-compatible dictionary. Observations determine molecular exposure; private
world data and transition `info` remain outside the controller's inputs.

## Coverage and source questions

`self_assembling_games.toylife.coverage.coverage()` returns 49 JSON-compatible
records: the 39 Barrett catalogue cases, six finite Skyrms cases, three constructed
multilocus cases and ECG. Each record has `id`, `support`, `tasks`, `assumptions`
and `requirements`. Task support describes an available interface. Successful
chemical behavior is an empirical result to establish for a particular genome,
apparatus, budget and evaluation distribution.

| Support | Cases |
|---|---|
| `native_task` | NAND, BOX, BAND, MH |
| `derived_task` | SIG, COD, HIER, CUE, MAN, ORD, TR-ORD, TR-PART, TR-LOG, SCOD, COMP-NAND, ADD, ADD-COMP, ASK, DIAL, DIAL-restricted, INQ, CHANGE, META-ORACLE, GEN, ATT-R, ATT-B, SK-SIG, SK-COMP, SK-VOTE, SK-DIALOGUE, ML-COORD, ML-TRANSFER, ML-COMPOSE |
| `analysis` | KIND |
| `needs_specification` | POLY, DIAL-costly, SUB, SET, RIT |
| `needs_organisation` | VIS, STAG, FFSR, DISC, INV, HYB, META-RULE, SK-TEAM, SK-CHAIN, ECG |

`native_task` means the target relation or choice problem is represented directly;
its source learner and historical protocol remain separately specified.
`derived_task` identifies an explicit reduction, such as replacing communicating
voters with noisy input fields. `needs_organisation` records interfaces,
participants or adaptation mechanisms needed to implement the case. `analysis`
identifies a measurement or interpretation to construct from an experiment.
For KIND, confusion metrics permit descriptive response partitions; the inventory
records the additional criterion needed to interpret those partitions as kinds.
GEN has a runnable adjacent/held-out ordering assay and is a `derived_task`.
`needs_specification` preserves a source question that affects its operational
contract. Every case has an individual assumption and requirement in the API.

The [catalogue](catalogue.md), [game sheets](games/signaling.md),
[finite dynamics](finite-dynamics.md), [multilocus assays](multilocus.md) and
[situated interaction guide](situated-interactions.md) retain the source-specific
contracts. Important open details include invention and forgetting, adjacent-pair
sampling, retained transfer templates, arithmetic ties and pretraining, costly
unlimited dialogue, subscription adaptation, meta-learning reset boundaries,
attention and distractor laws, primitive salience dynamics, Skyrms costs and
dialogue information sets. The coverage inventory makes these requirements
visible alongside the assays that can exercise related chemical capacities.

## Experimental comparisons

Frozen assessment accepts `system.evaluate(panel=1000000, episodes=100)` for
an episodic task. Choose evaluation panels outside the training generation range
and reserve them before selecting genomes. The episode override leaves training
state and its configured evaluation budget unchanged. Static tasks enumerate
cases exactly and reject an episode override.

World draws are shared across candidate genomes within each panel and episode;
action randomness has a stable genotype address. Duplicate genomes therefore
receive the same lifetime outcome. Population size counts reproductive copies;
independent episodes, panels and population runs supply the units for uncertainty.
The evaluation's `sampling` record makes this convention explicit.

Use several declared molecular panels to measure apparatus sensitivity. Rank
identities supplied through one-hot encoding have chemical similarities that
need not follow their numerical order. Dose encoding supplies a different physical
organisation and should be a separate comparison condition. A phase reset also
resets basal chemical development; compare age-matched blank and inverted-cue
delay controls when attributing performance to information retained from a cue.

The native mutation probability is per offspring: a selected offspring receives
one uniformly chosen bit flip with that probability. It is not a per-bit rate.
At fixed probability, increasing gene count spreads the same mutation-event rate
across more sites. Retain this convention in comparisons with multilocus mutation.
