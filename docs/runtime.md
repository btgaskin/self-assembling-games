# Runtime guide

The package provides a serial reinforcement engine, a Python API, and the `sg` command. It keeps the whole book catalogue discoverable and distinguishes executable demonstrations from source-reproduction profiles.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/sg list
.venv/bin/sg list --all
.venv/bin/sg describe HIER
.venv/bin/sg run hier.demo --seed 0 --plays 10000 --output runs/hierarchy
.venv/bin/sg inspect runs/hierarchy
.venv/bin/sg evaluate runs/hierarchy
.venv/bin/sg resume runs/hierarchy --plays 10000
.venv/bin/python -m pytest -q
```

`run` chooses a fresh output directory; `resume` advances a checkpoint. Python `Simulation.run(plays=N)` also advances by N additional interactions. The default profile duration applies when no duration is given. For visiting/inquiry models, one interaction is one visit/consultation; round and problem clocks report their larger units.

```python
from self_assembling_games import build, evaluate, load_profile

profile = load_profile("sig.n2.demo")
simulation = build(profile, seed=0)
simulation.run(plays=10_000)
assessment = evaluate(simulation.snapshot())
simulation.checkpoint("runs/signaling")
```

Profiles are JSON-serializable records. Save `load_profile(name).to_dict()` to inspect or adapt a configuration, then load that JSON path through the CLI or API. Mandatory distributions, bindings, schedules, costs and source gaps are validated. Unsupported game/learner combinations are rejected with a concrete message.

The default weight is one per option. Basic state/signal/act identities are integer indices. HIER properties use the order black dog, black cat, white dog, white cat: color is `state//2`, animal is `state%2`; context zero/one/two requests color/animal/both. Named modules and table records expose the interpretation. Signal tokens remain sender-specific.

## Families and profile kinds

The model register has thirty-nine entries. Executable families include signaling, coding, NAND, hierarchy, cue reading, manipulation, invented vocabularies, template transfer, computation, finite/costly discourse, visiting networks, inquiry, bandits, nested learning, and attention.

Demonstrations requiring choices beyond the book label them explicitly: pair-row failure handling, the frozen-vocabulary export policy, FFSR decay/error timing, bounded propensities, distractor laws, dialogue history/costs, and Monty Hall host behavior. ATT-B has distinct `source_value` and `value` profiles.

R4 is the literal per-unchosen-option ARP update on a single-vector bandit/adviser task. Its multi-urn canonical scope remains gated. Whole-strategy WSLR and act-based R6 retain deterministic assignments separately from propensity weights. KIND is available as `conventional_kinds(snapshot, module, tolerance=0)`; a positive tolerance groups contexts against each group's first representative.

The source-specific four-state R1 and eight-state punishment configurations are `sig.n4.book` and `sig.n8.punishment.book`. Their horizons are one million and ten million plays. `sg list --experiments` records the book's reference thresholds and outstanding protocol fields. Registering them does not execute those experiments.

## Frozen modules and learned transfer

Constructed perfect fixtures validate routing and translation. To transfer through an actually learned NAND module:

```python
from self_assembling_games import build, load_profile

template = build("nand.demo", seed=3)
template.run(plays=10_000)
path = template.checkpoint("runs/pretrained-nand")

profile = load_profile("tr-log.demo")
profile.parameters.update(
    template_mode="artifact",
    templates={"template": str(path)},
)
transfer = build(profile, seed=4)
transfer.run(plays=10_000)
```

COMP-NAND artifact maps use `first` and `second` keys. TR-ORD/TR-PART use `ordering`; ADD-COMP uses `addition` and `comparison`. The latter needs an eleven-stimulus ordering artifact and an addition artifact. Saved templates are copied into distinct frozen instances, preserving their stochastic policies. Lineage includes original configuration, seed, training duration, metrics and artifact hash.

An invented vocabulary contains a mutator and may lack receiver rows for unseen pairs. Its fixed-template export therefore requires explicit choices:

```python
from self_assembling_games import build, export_template

ordering = build("ord.demo", seed=5)
ordering.run(plays=10_000)
path = export_template(
    ordering, "runs/ordering-template",
    suppress_mutator=True,
    complete_pair_rows=True,
)
```

These transforms preserve the original simulation, record their lineage, and initialize missing pair rows uniformly. Stored training metrics describe the original run; evaluate the exported frozen policy separately. The export checks a default budget of ten thousand possible pair rows; a larger budget must be explicit. The canonical interpretation of these choices remains Q12. A frozen vocabulary with a live mutator or missing pair rows is rejected rather than changed during transfer. Pretraining budget and checkpoint choice are supplied by the experiment, and their cost is separately visible.

Use `held_out_ordering(snapshot, samples=..., seed=...)` or `sg evaluate --protocol held-out` to test nonadjacent comparisons without the adjacent-only training sampler or updates.

## Evaluation and measurement

Exact evaluation enumerates the same finite interaction function under forced chance branches. This preserves missing-bit completion, current persistent assignments, and shared stochastic ancestors. It applies no learning and consumes no training randomness. A branch limit bounds its computational cost.

Stateful network/inquiry, nested tasks, and invention use sampled frozen evaluation. Sampled evaluation advances a copied environment with fixed adaptive state and a separate random generator. It measures that frozen rollout; learning-rule performance comes from training experiments.

Metrics include attempts, completed outcomes, cumulative accuracy, bounded-window accuracy, net payoff, terms, exchanges, and outcome statuses. Censored conversations and resource stops have explicit status counts and are excluded from completed-outcome denominators. Exact evaluation reports completed and truncated probability mass separately.

Finite dialogue ends in failure if the assistant asks a third question. ASK/costly demonstrations instead use an explicit twenty-exchange engineering cap; reaching it records censoring and supplies no update.

## Ensembles, checkpoints, and traces

```sh
.venv/bin/sg ensemble sig.n2.demo --seeds 0 1 2 3 --plays 10000 \
  --threshold 0.95 --workers 2 --output runs/ensemble
.venv/bin/sg benchmark sig.n2.demo --plays 10000
.venv/bin/sg run nand.demo --plays 100 --output runs/debug \
  --trace runs/debug/events.jsonl
.venv/bin/sg replay runs/debug/events.jsonl
```

The ensemble fixes its seed set, duration and cumulative-accuracy cutoff. Repeating the same command resumes incomplete checkpoints and reuses complete runs; conflicting protocols are rejected. Independent processes own separate seed artifacts and preserve serial results. Reports include the run distribution, mean accuracy with a percentile bootstrap interval, and threshold frequency with a Wilson interval.

Checkpoints preserve table visitation order and contain all tables, assignments, reference points, clocks, world/interaction state, RNG state, novelty IDs, lineage and measurement state. JSON round-trips retain float64 weights. Writes use atomic replacement. The runtime records package/NumPy versions, repository revision and a package-source fingerprint; continuation requires the pinned implementation. Learned template imports retain old provenance while copying policy state into the current runner.

Default runs retain bounded measurement windows and no complete event history. Enabling `runtime.record` retains only the current interaction. Full traces stream chance, decision, feedback, structural, assignment and outcome events with a starting checkpoint. Replay verifies the event sequence and final state; it currently loads the explicitly requested debug log into memory.

Structural history includes explicit fixture changes and the iterative phase schedule; clocks reconstruct the deterministic phase progression. Belief resets and other transitions appear in full logs. Benchmark results measure the current serial implementation; they establish no compiled-backend speedup.

## Validation

The suite covers every registered profile, source gates, numeric updates, finite-state counts, exact fixtures, signed successful payoffs, unsampled reciprocity, stochastic fan-out, repeated draws, asynchronous inquiry, dynamic rollback/extinction, frozen artifacts, held-out evaluation, inner resets, event replay, resumable ensembles, independent-process parity, and CLI behavior.

The [qualification report](qualification.md) records the actual test, build, and benchmark results. Long book-horizon statistical reproduction is a separate registered experiment.
