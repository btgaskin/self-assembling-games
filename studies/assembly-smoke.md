# Assembly extension: retained smoke studies

Seven declared comparison groups exercise acquired-state release, component reuse,
joint hierarchical learning, cross-context reinforcement and divergent utilities.
Each condition uses seeds 0, 1 and 2. All endpoint evaluations below are exact.
These short runs measure early dispositions and check integration; their small seed
set and horizon support exploratory comparisons.

The [machine-readable report](assembly-smoke-results.json) retains each endpoint,
verification result, run path, source hash and elapsed time. The [runner](assembly_smoke.py)
uses the archived measurements, and every run retains its original and resolved TOML,
checkpoint states, receipts and numerical outputs. The [interpretation](assembly-interpretation.md)
connects these results and explains the additional release control. A [compressed run archive](assembly-smoke-runs.tar.gz)
contains the full retained directories; its SHA-256 is recorded in the JSON report.

| Study | Trials | Wall time (s) | Verified |
|---|---:|---:|---|
| assembly-coadaptation | 9 | 39.78 | True |
| assembly-conflict | 33 | 15.47 | True |
| assembly-generalization | 18 | 10.11 | True |
| cue-release | 18 | 3.30 | True |
| man-release | 18 | 2.03 | True |
| nand-compose | 18 | 121.07 | True |
| nand-transfer | 15 | 4.82 | True |

Peak resident memory across the reporting processes: 84.1 MiB.
Timing includes running, retaining and verifying the study. The composed cases include
exact enumeration of both stochastic components at each measurement.

## Endpoint comparisons

Success is the declared task predicate. For generalization it means exact action/state
agreement; Gaussian reward also credits nearby actions. Conflict utilities are reported
separately. ODE comparisons end at time 1; learning comparisons use their declared encounter
budgets. The range is across three seeds, not a confidence interval.

### assembly-coadaptation

[Configuration](../runs/assembly-coadaptation/experiment.toml) · [Retained run](../runs/assembly-coadaptation/20261015T101703.407764Z-3166e894)

| Condition | Mean success | Seed range |
|---|---:|---:|
| stipulated | 0.2015 | 0.0463–0.3048 |
| role-free | 0.2349 | 0.1922–0.3154 |
| observation-control | 0.1921 | 0.0463–0.2896 |


### assembly-conflict

[Configuration](../runs/assembly-conflict/experiment.toml) · [Retained run](../runs/assembly-conflict/20261015T101743.225409Z-82673c7c)

| Condition | Mean success | Seed range |
|---|---:|---:|
| gm16.table1.r1 | 0.2508 | 0.1880–0.3184 |
| gm16.table1.evolution | 0.3273 | 0.3187–0.3332 |
| zbh13.separating.signed | 0.8245 | 0.8046–0.8477 |
| zbh13.separating.evolution | 0.5206 | 0.4183–0.5930 |
| zbh13.hybrid-equilibrium.signed | 0.6947 | 0.6378–0.7439 |
| zbh13.hybrid-equilibrium.evolution | 0.5399 | 0.4375–0.6335 |
| scod.conflict.lambda-0.r1 | 0.3453 | 0.3210–0.3773 |
| scod.conflict.lambda-0.25.r1 | 0.3138 | 0.2846–0.3519 |
| scod.conflict.lambda-0.5.r1 | 0.2902 | 0.2677–0.3298 |
| scod.conflict.lambda-0.75.r1 | 0.2665 | 0.2392–0.2806 |
| scod.conflict.lambda-1.r1 | 0.2545 | 0.2376–0.2720 |

- gm16.table1.r1: mean participant utilities receiver=4.7902, sender=4.0692.
- gm16.table1.evolution: mean participant utilities receiver=4.1614, sender=3.0059.
- zbh13.separating.signed: mean participant utilities receiver=0.8245, sender=0.1099.
- zbh13.separating.evolution: mean participant utilities receiver=0.5206, sender=0.0786.
- zbh13.hybrid-equilibrium.signed: mean participant utilities receiver=0.6947, sender=0.0089.
- zbh13.hybrid-equilibrium.evolution: mean participant utilities receiver=0.5399, sender=0.2691.
- scod.conflict.lambda-0.r1: mean participant utilities A=0.3453, B=0.3453, receiver=0.3453.
- scod.conflict.lambda-0.25.r1: mean participant utilities A=0.2892, B=0.2938, receiver=0.3138.
- scod.conflict.lambda-0.5.r1: mean participant utilities A=0.2579, B=0.2651, receiver=0.2902.
- scod.conflict.lambda-0.75.r1: mean participant utilities A=0.2491, B=0.2535, receiver=0.2665.
- scod.conflict.lambda-1.r1: mean participant utilities A=0.2426, B=0.2583, receiver=0.2545.

### assembly-generalization

[Configuration](../runs/assembly-generalization/experiment.toml) · [Retained run](../runs/assembly-generalization/20261015T101758.660858Z-0677ab8d)

| Condition | Mean success | Seed range |
|---|---:|---:|
| binary-alpha-0 | 0.8860 | 0.7779–0.9560 |
| binary-alpha-0.1 | 0.7132 | 0.6134–0.8237 |
| binary-alpha-1 | 0.5000 | 0.5000–0.5000 |
| gaussian-width-0 | 0.0602 | 0.0548–0.0647 |
| gaussian-width-0.5 | 0.0606 | 0.0558–0.0631 |
| gaussian-width-2 | 0.0723 | 0.0677–0.0755 |

- binary-alpha-0: mean expected reward 1.8860; normalised reward 0.9430.
- binary-alpha-0.1: mean expected reward 1.7132; normalised reward 0.8566.
- binary-alpha-1: mean expected reward 1.5000; normalised reward 0.7500.
- gaussian-width-0: mean expected reward 0.2545; normalised reward 0.2545.
- gaussian-width-0.5: mean expected reward 0.2599; normalised reward 0.2599.
- gaussian-width-2: mean expected reward 0.3086; normalised reward 0.3086.

### cue-release

[Configuration](../runs/assembly-lifecycle/cue-release/experiment.toml) · [Retained run](../runs/assembly-lifecycle/cue-release/20261015T102210.780168Z-007779fe)

| Condition | Mean success | Seed range |
|---|---:|---:|
| release | 0.9727 | 0.9725–0.9728 |
| release-acquired-side-frozen | 0.9623 | 0.9617–0.9631 |
| fixed-side-retained | 0.9898 | 0.9896–0.9899 |
| fresh-online-budget | 0.8469 | 0.7626–0.9262 |
| fresh-total-budget | 0.9194 | 0.8749–0.9657 |
| release-without-acquisition | 0.9529 | 0.9363–0.9620 |


### man-release

[Configuration](../runs/assembly-lifecycle/man-release/experiment.toml) · [Retained run](../runs/assembly-lifecycle/man-release/20261015T102214.053790Z-5fefd630)

| Condition | Mean success | Seed range |
|---|---:|---:|
| release | 0.9727 | 0.9725–0.9730 |
| release-acquired-side-frozen | 0.9623 | 0.9615–0.9631 |
| fixed-side-retained | 0.9898 | 0.9896–0.9899 |
| fresh-online-budget | 0.8469 | 0.7626–0.9262 |
| fresh-total-budget | 0.9194 | 0.8749–0.9657 |
| release-without-acquisition | 0.9601 | 0.9580–0.9614 |


### nand-compose

[Configuration](../runs/assembly-lifecycle/nand-compose/experiment.toml) · [Retained run](../runs/assembly-lifecycle/nand-compose/20261015T101811.244537Z-08d6d812)

| Condition | Mean success | Seed range |
|---|---:|---:|
| learned-short-one | 0.5553 | 0.5317–0.5676 |
| learned-long-one | 0.6832 | 0.6413–0.7390 |
| learned-long-two | 0.6226 | 0.6014–0.6408 |
| learned-long-one-uniform-route | 0.6753 | 0.6413–0.7108 |
| constructed-perfect-1-selector | 0.8840 | 0.8111–0.9356 |
| constructed-perfect-2-selector | 0.8169 | 0.7130–0.8749 |


### nand-transfer

[Configuration](../runs/assembly-lifecycle/nand-transfer/experiment.toml) · [Retained run](../runs/assembly-lifecycle/nand-transfer/20261015T102012.292368Z-3052735a)

| Condition | Mean success | Seed range |
|---|---:|---:|
| learned-short | 0.5936 | 0.5238–0.6303 |
| learned-long | 0.7187 | 0.7137–0.7229 |
| constructed-perfect | 0.8280 | 0.7304–0.8888 |
| scratch-online-budget | 0.7336 | 0.7134–0.7562 |
| scratch-total-long-budget | 0.7650 | 0.7321–0.7829 |

