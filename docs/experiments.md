# TOML experiments and workspaces

An experiment declares resolved models, named conditions, independent replicate
seeds, a budget for each condition, and analysis settings. Loading TOML resolves
its defaults and validates declarations. Running constructs seeded systems and
archives the whole study under one timestamped directory.

Schema 2 describes independent single-model trials. Schema 3 adds ordered phases
whose starting states can be fresh systems, descendants of an earlier phase, or
fresh systems assembled from explicitly exported components.

Install the package with its runtime dependencies, including `tomli-w`, then:

```sh
sg init experiments/example --model sig.demo --budget 100
sg resolve experiments/example/experiment.toml
sg run experiments/example/experiment.toml --workspace example
sg status runs/example/<timestamp>
sg evaluate runs/example/<timestamp>
sg verify runs/example/<timestamp>
sg resume runs/example/<timestamp> --additional-budget 100
```

`sg catalogue` lists games and executable presets with source and implementation
metadata. `sg describe sig.demo` and `sg resolve sig.demo` show a resolved model.
All commands print JSON. `verify` returns exit code 1 for changed files;
configuration errors return exit code 2 and trial failures return exit code 1. A completed study can
contain failed trials, recorded with `status = "complete_with_failures"` in its
manifest and a separate error for each affected seed.

## Compact declarations

The smallest experiment is:

```toml
preset = "sig.demo"
budget = 100
seeds = [0, 1, 2]
```

The omitted values are `schema_version = 2`, `kind = "experiment"`,
`seeds = [0]`, `workers = 1`, and empty analysis/protocol tables. An omitted
budget uses each resolved model's declared execution budget. Discrete encounter
and generation budgets are nonnegative integers; continuous time accepts finite
nonnegative numbers. Seeds are distinct nonnegative integers.

Use shared overrides and conditions for comparisons:

```toml
schema_version = 2
kind = "experiment"
preset = "sig.demo"
seeds = [10, 20, 30]
budget = 100
workers = 2

[overrides.execution.parameters]
window = 25

[[conditions]]
name = "baseline"

[[conditions]]
name = "short-window"
budget = 50

[conditions.overrides.execution.parameters]
window = 10

[analysis]
sample_interval = 10
bootstrap_samples = 1000
bootstrap_seed = 7
confidence = 0.95
threshold_metric = "success_probability"
threshold = 0.9

[analysis.evaluation]
protocol = "exact"
```

A condition inherits the top-level preset, budget and protocol. It can choose
its own `preset` and `budget`. Resolution applies preset defaults, then the
shared `overrides`, then `conditions.overrides`. Nested tables merge recursively;
arrays and scalar values replace earlier values. Per-condition protocol entries
replace shared protocol entries by the same rule. `model` is an alias for
`preset`; choose one field per declaration. Each condition name must be unique.
Unknown declaration fields produce an error.

Overrides use the model's scientific components: `interaction`, `organisation`,
`realisation`, `adaptation`, `source`, `execution`, and `resources`. For example,
`[overrides.adaptation.parameters]` configures the local adaptation rule. Use
`sg resolve` to inspect the exact supported fields and the full source declaration
before running. `protocol` contains execution controls accepted by the selected
runtime; budget belongs to the experiment condition.

To save a reusable model rather than a study:

```toml
kind = "model"
preset = "sig.demo"

[overrides.execution.parameters]
window = 20
```

Python and TOML share the same declarations:

```python
from self_assembling_games import model, experiment, load, save, run
from self_assembling_games.experiments import Workspace

specification = model("sig.demo")
save(specification, "models/signalling.toml")
assert load("models/signalling.toml").to_dict() == specification.to_dict()
study = experiment(specification, seeds=[10, 20], budget=100)
result = run(study, workspace="example")  # runs/example/<timestamp>
result.verify()
result.evaluate()  # evaluates restored checkpoint copies
result.resume(additional_budget=100)

# Explicit workspace roots use precisely the supplied directory.
Workspace("runs/another-study").run(study)
```

`save` uses the maintained `tomli-w` writer. Resolved TOML preserves optional
Python nulls through the reserved top-level `toml_nulls` array of JSON pointers;
`load` restores those values before resolving the declaration. Compact human
written TOML usually needs no such entries.

## Retained files and continuation

`run(..., directory="runs", workspace="example")` writes
`runs/example/<UTC timestamp>-<unique suffix>/`. Omitting `workspace` writes a
study directly under `directory`. A `Workspace(path)` uses that exact path as
its study root. Parallel workers execute independent condition/seed trials in
separate processes; trial order in the results remains declaration/seed order.

Each study retains:

- `config.toml`: the original input, including comments when supplied as a file.
- `resolved.toml` and `resolved.json`: every resolved condition, seed, budget,
  protocol and analysis setting.
- `manifest.json`: software identity, trial status, paths, positions, file hashes
  and the history of added budgets.
- `trials/<condition>/seed-<seed>/checkpoint.json`: the latest complete System
  checkpoint, including its resolved model, controller state and RNG state.
- Per-trial `results.json`, and `trajectory.json` when sampling is requested.
- Study `results/summary.json`: ordered trial results and descriptive replicate
  summaries, also available as `results.json` for programmatic compatibility.

`resume(path)` finishes the archived budget for incomplete trials. A failed
measurement can resume from its retained endpoint without replaying training.
`resume(path, additional_budget=100)` adds 100 clock units to every trial's
archived target; the original resolved configuration stays intact and the
manifest records the extension. Resume reads archived models and checkpoints;
it does not resolve the current preset registry. Checkpoint verification also
checks runtime/source identity, so continuation uses the matching implementation.

The runner checkpoints before execution and after every requested sampling
interval. With endpoint-only analysis, an interruption during an advance can
replay the unfinished advance from its last retained checkpoint. Evaluation uses
restored copies and preserves the training RNG trajectory. File hashes and
checkpoint identities make changed or missing artifacts visible through
`verify`; continuation rejects a failed verification.

## Analysis

Every successful trial retains its endpoint evaluation separately. Optional
`sample_interval` evaluates at position zero and each interval on the runtime's
clock; intermediate samples are retained as a trajectory. A final endpoint
between intervals remains in `results.json`. Sampling a continuous ODE changes
solver integration boundaries, so compare conditions with the same sampling
schedule when numerical replay matters.

Built-in evaluation defaults to exact measurements when supported, otherwise to
sampled measurements. Situated evaluation summarizes retained encounter outcomes.
An explicit evaluation protocol remains part of the resolved study declaration.

For each numeric endpoint metric, the study records replicate count, mean,
sample standard deviation, minimum, maximum, and a percentile bootstrap interval
for the replicate mean. The bootstrap resamples independent seeds using
`bootstrap_seed` and `bootstrap_samples`; it uses `confidence` for interval
coverage. Defaults are seed 0, 1000 draws and 0.95 coverage. A single replicate
has no standard deviation or bootstrap interval. Failed trials have separate
counts and error records and do not contribute to metric summaries.

With `threshold` and `threshold_metric`, each measured trial records whether any
retained sample or endpoint reached the threshold, its first observed position,
and whether the budget ended before an observed crossing. Study summaries give
reached and budget-censored counts and a Wilson interval for the reached
proportion. These are observations on the sampling schedule; a crossing between
observations can occur earlier. Runtime terminal statuses remain visible
alongside failures and budget censoring. The underlying evaluation retains any
truncation or incomplete-mass fields supplied by that runtime.

## Ordered phases

This declaration acquires a cue response, releases the sender into mutual
signalling, and branches from the same acquired endpoint with a frozen receiver:

```toml
schema_version = 3
seeds = [0, 1, 2]

[[phases]]
id = "cue"
preset = "cue.demo"
budget = 100

[[phases]]
id = "mutual"
preset = "sig.demo"
budget = 100
[phases.start]
kind = "carry"
phase = "cue"
[phases.start.transition]
kind = "release_fixed"
fixed_weight = 9.0
other_weight = 1.0

[[phases]]
id = "frozen-receiver"
preset = "sig.demo"
budget = 100
[phases.start]
kind = "carry"
phase = "cue"
[phases.start.transition]
kind = "release_fixed"
fixed_weight = 9.0
other_weight = 1.0
[[phases.interventions]]
kind = "trainability"
modules = ["receiver"]
enabled = false

[analysis]
sample_interval = 25
```

These budgets are illustrative configuration choices. The release distribution
is an explicit experimental intervention: its old fixed option receives weight
9 and each other option receives weight 1, then proportional R1 becomes active.
The acquired receiver, world, random generator, native clocks and accumulated
measurements carry into both descendants. Each descendant owns a separate copy.
An `identity` transition carries the same scientific model without changing its
state. Supported transitions validate their observation/action domains,
ownership, adaptation and execution schedule.

Top-level `phases` gives a single condition named `default`. Multiple conditions
use `[[conditions]]` with nested `[[conditions.phases]]`. Every phase has a
unique `id` within its condition. `preset`, `model`, `overrides`, `budget`,
`protocol` and `analysis` inherit from the experiment and condition; phase values
take precedence. Resolved phases retain `model`, `budget`, `start`, `protocol`,
`interventions` and `analysis`. References name earlier phase ids in the same
condition and seed trial. A missing `start` means `kind = "fresh"`.

The first carry implementation supports the built-in conditional-table runtime.
Identity carry preserves its interaction; fixed-side release supports CUE/MAN → SIG.
Trainability interventions apply to the admitted fixed-shape games, and component
exports support NAND, ORD and ADD interfaces. Fresh phases can use the other
registered runtimes; their state transformations require a provider-specific contract.

Fresh and composed phases use the first eight bytes of
`SHA256(JSON([trial_seed, phase_id]))`, interpreted as an unsigned big-endian
integer, as their seed. The JSON encoding uses no separator whitespace. This
rule preserves fresh phase seeds under condition reordering and worker changes.
Carry phases retain their parent's generator state and seed. An explicit carry
protocol must equal the retained System protocol; changing numerical controls
during a carry is currently unsupported.

Python accepts the same declarations through `config.from_dict(...)` or:

```python
from self_assembling_games import experiment, run

study = experiment("sig.demo", seeds=[2, 5], phases=[
    {"id": "acquire", "budget": 100},
    {"id": "continue", "budget": 50,
     "start": {"kind": "carry", "phase": "acquire",
               "transition": {"kind": "identity"}}},
])
result = run(study, workspace="phases")
```

### Component composition

A composed phase starts a new target system using immutable exports of named
earlier endpoints. For example:

```toml
schema_version = 3
seeds = [3]

[[phases]]
id = "nand"
preset = "nand.demo"
budget = 100

[[phases]]
id = "transfer"
preset = "tr-log.demo"
budget = 100
[phases.start]
kind = "compose"
[phases.start.bindings.template]
phase = "nand"
```

Each binding names a destination template slot, an earlier `phase`, and an
optional `export` table with the component export specification. The target's
effective model uses `template_mode = "artifact"` and binds archived component
paths in `interaction.parameters.templates`. The report retains both the
declared model hash and this effective model, the export specification, source
endpoint receipt hash and component file hash. Module pretraining remains a
separate measured phase. Exporting an invented vocabulary requires explicit
choices for mutator suppression and completion of unseen receiver pairs.

### Retained state and continuation

Phase directories contain immutable numbered checkpoint generations. Each
`receipt.json` links to the previous receipt and hashes its generation's
`checkpoint.json`. A receipt records native and phase positions, the actual
starting position, initialization lineage, effective model, trainability
interventions, measurement and pending-measurement status. The final endpoint
is preserved alongside all earlier committed generations. Reports give both
native `position` and local `phase_position`; carrying a system preserves its
native clock rather than resetting a learner's internal phase schedule.

The receipt is committed after the checkpoint. A hard stop can leave an
unfinished generation, which recovery ignores, or a stale `cursor.json` or
manifest, which recovery reconstructs from complete receipts. A failed
measurement is retried at its retained boundary before any further advance.
Every evaluated endpoint is retained, including endpoints between sampling
intervals, so a later continuation preserves observed threshold crossings.

`resume(path)` completes declared phases. For schema 3,
`resume(path, additional_budget=N)` extends only the final phase in each
condition. Its earlier endpoints and all ancestor phases remain immutable; the
extension receives a hash-linked journal entry with scope `final_phase`.
Budgets remain additional native clock units within the phase. Different
conditions can use different final-phase clocks, so interpret their budgets on
those clocks. Verification checks configuration hashes, checkpoint/receipt
chains, component hashes and parent endpoint identities. Cursors and report
files are derived views that can be regenerated after an interrupted write.

Schema 2 runs also retain immutable recovery checkpoints and boundary
measurements. Their existing `trajectory.json` continues to contain interval
samples; off-grid endpoints and pending measurements are additionally retained
in the recovery journal. The existing schema 2 extension semantics still add
budget to every independent trial.

For schema 3 studies, `sg run ... --budget N` overrides each condition's final-phase budget. Earlier acquisition budgets stay as declared. `--seeds`
replaces the trial seed set.
