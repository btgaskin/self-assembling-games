# Assembly lifecycle studies

These four schema 3 workspaces declare bounded pilot comparisons of precursor
coordination, acquired-module transfer and computational composition. They fix
their seed sets, phase budgets, export endpoints and measurement schedules
before execution. Loading each TOML resolves its declarations without running
the study. The budgets and four-seed samples are starting experimental choices;
these files contain hypotheses and protocols, with results retained in the
timestamped directories created by `sg run`.

| Workspace | Acquired organization | New adaptive relation |
|---|---|---|
| [cue-release](cue-release/experiment.toml) | Receiver responses to a fixed sender | Mutual signalling after releasing the sender |
| [man-release](man-release/experiment.toml) | Sender choices acting on a fixed receiver | Mutual signalling after releasing the receiver |
| [nand-transfer](nand-transfer/experiment.toml) | A learned stochastic NAND module | New input translators producing OR |
| [nand-compose](nand-compose/experiment.toml) | Two independently learned stochastic NAND modules | Selection of component outputs as another component's inputs |

## What the source supplies

Barrett's cue-reading and sensory-manipulation precursors fix one side while
the other learns. His ritualization discussion motivates subsequent mutual
adaptation. The exact release time and conversion from a fixed disposition to
plastic propensities are declared here as experimental variants.
([Barrett, chapter 4, pp. 63–68](https://doi.org/10.1093/9780197793572.003.0004);
[repository specification](../../docs/games/transfer.md).)

Template transfer reuses the learned dispositions of a frozen game through new
input translators. NAND-to-OR supplies the small complete truth-table domain
used here. Composition then learns which supplied source should feed an input
of a frozen component.
([Barrett, chapter 5, pp. 89–92](https://doi.org/10.1093/9780197793572.003.0005);
[chapter 7, pp. 112–116](https://doi.org/10.1093/9780197793572.003.0007);
[repository computation specification](../../docs/games/computation-discourse.md).)

All action alphabets, observations, permitted sources and success criteria are
given. The acquired policies and their coupling are the adaptive objects under
study. The source register retains unresolved historical-protocol questions;
these pilot declarations use the explicit demonstration and variant choices of
their resolved models.

## Cue and manipulation release

Each workspace uses four seeds, uniform two-state nature and proportional R1.
Precursor acquisition lasts 1,000 encounters. At release, the formerly fixed
option receives propensity 9 and its other option receives propensity 1; its
fixed assignment is cleared. Its acquired partner retains all learned state.
Both now receive the ordinary SIG feedback. This finite release bias gives the
old disposition a 0.9 initial response probability with room for adaptation.

**Hypothesis:** acquired coordination changes the subsequent acquisition and
stability of mutual signalling, compared with a fresh pair. Allowing both sides
to adapt may produce a different recovery trajectory from retaining the
acquired side's policy while only its formerly fixed partner adapts.

The declared arms are:

- `release`: acquire the precursor, release its fixed side, train both sides for
  1,000 additional encounters.
- `release-acquired-side-frozen`: use the same acquired precursor and release
  weights; freeze the acquired receiver in CUE or acquired sender in MAN during
  the 1,000-encounter test.
- `fixed-side-retained`: continue the original precursor for 1,000 encounters,
  retaining its original fixed side. This measures performance under the
  preserved relationship supplied by the precursor.
- `fresh-online-budget`: train a fresh SIG pair for 1,000 encounters, matching the
  online budget after release.
- `fresh-total-budget`: train a fresh SIG pair for 2,000 encounters, matching the
  precursor plus release training budget.

The shared `precursor` phase id and seed label reproduce the same acquired state
across the three precursor arms. These are paired counterfactuals on an acquired
system. Fresh SIG starts use their separately declared phase seed. Give both
the observed release cost and subsequent recovery; a high endpoint alone does
not identify which side reorganized. Stored table dispositions support
convention and participant-specific change analyses.

## Acquired NAND transfer

**Hypothesis:** the acquired template's competence constrains downstream OR
transfer, while a useful template can reduce additional acquisition relative to
learning the target task directly.

`learned-short` and `learned-long` export the complete NAND policy after fixed
budgets of 100 and 1,000 encounters, then train only its new translators for
1,000 encounters. The two conditions share the `nand` phase id and seed: the
short checkpoint is a prefix of the longer run. Every declared seed contributes;
there is no selection of a successful checkpoint or seed. Inspect the actual
source-policy evaluations before treating training duration as template quality.

`constructed-perfect` uses the existing perfect NAND fixture and the same
1,000-encounter transfer budget. It exercises translator learning under supplied
component competence. `scratch-online-budget` learns OR directly using the NAND
kernel with truth table `[0,1,1,1]` for 1,000 encounters. Its two senders observe
the same binary target inputs; its receiver learns a response to their signals.
`scratch-total-long-budget` uses 2,000 encounters, matching the 1,000 encounters
of long pretraining plus 1,000 encounters of transfer. Its observation and update
architecture remains recorded separately from the translator architecture.

Compare conditional online savings and total-budget performance separately. A
perfect fixture supplies an initial capacity; it does not supply a measured
pretraining bill.

## Acquired NAND composition

**Hypotheses:** component quality and the number of adaptive input selectors
jointly affect the acquisition of the composite function; selecting a useful
route provides a benefit over maintaining an initially uniform route.

The target is `(X NAND Y) NAND Q`, with independent uniform binary X, Y, Q and
an irrelevant candidate Z. Each learned arm first trains independent `nand-first`
and `nand-second` systems. Distinct phase ids give those modules distinct fresh
seeds. The `short` arm trains each for 100 encounters; the `long` arms train each
for 1,000. The selected fixed-budget endpoints are exported and installed into
distinct frozen instances. Their internal policies preserve the source's
stochastic responses. The downstream phase lasts 1,000 encounters.

The one-selector arms learn whether module 2 should receive Z or module 1's
output. The two-selector arm additionally chooses between Q and module 1's
output at the other input. `learned-long-one-uniform-route` freezes P's initial
uniform policy, supplying an adaptive-routing control with the same acquired
components. Constructed-perfect one- and two-selector arms retain the existing
fixture comparisons.

This workspace studies acquired frozen components and adaptive coupling. COMP's
current feedback credits the selectors; a study of adaptive component internals
would require a declared additional credit rule. The available kernel has no
three-input direct scratch comparator, so this workspace does not estimate
total-budget superiority to learning the composite function from scratch.

## Measurements and execution

The primary measurement is exact frozen-policy success on each task's full
distribution, at phase position zero, each 100 encounters and the endpoint.
Reports retain the per-seed trajectory, endpoint distribution and first observed
crossing of 0.9, with budget censoring. Four independent seeds support a pilot
description; the bootstrap intervals use 1,000 draws with seed 7. They are not a
predeclared powered confirmatory design.

Retain acquired-module quality, router dispositions, frozen/trainable inventory,
and source artifact hashes alongside task success. Training accuracy within a
carry phase can be calculated from the differences in retained completed and
successful outcome counters between its initial and final checkpoints. Report
pretraining and downstream encounters separately and sum them for comparisons
that explicitly match total acquisition exposure. Participant/module events
within one seeded trial are trajectory observations.

```sh
sg resolve runs/assembly-lifecycle/cue-release/experiment.toml
sg run runs/assembly-lifecycle/cue-release/experiment.toml --workspace assembly-lifecycle/cue-release
sg verify runs/assembly-lifecycle/cue-release/<timestamp>
```

The other workspaces use the same commands with their paths. A lifecycle smoke
check may resolve a copy with small budgets; retain that changed declaration and
label its output as a mechanics check. `resume` finishes interrupted phases.
Schema 3 `additional_budget` extends only the final phase of each condition;
the precursor endpoints and exported components remain immutable. See the
[phase workflow](../../docs/experiments.md#ordered-phases) for the receipt and
continuation contracts.
