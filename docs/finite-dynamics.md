# Finite games and adaptive methods

`FiniteGame` describes permitted decision contexts, option sets, chance events, participant returns, and interaction timing. It compiles the interaction into chance-weighted leaves and complete deterministic role plans. A plan assigns an option to every adaptive context, including contexts an encounter may leave unvisited. Frozen conditional components remain explicit chance distributions. Exact expectations sum over these leaves; role-plan compatibility masks avoid constructing a dense multi-player payoff tensor.

`AdaptiveProfile` combines a game with a method, population structure, protocol, resource budgets, and optional pretrained lineage. `register_finite_game` supplies additional source-labelled factories through the same interface. `build`, `load_profile`, `resume`, and `evaluate` dispatch between the existing Barrett encounter runtime and the finite runtime. `finite_catalogue` lists games with finite adapters; the original `catalogue` retains the Barrett chapter catalogue.

## Evolution

The default mathematical method is `replicator.continuous`. For population frequencies `x_i`, random-matching expected fitness `f_i`, and mean fitness `sum_i x_i*f_i`,

```text
dx_i/dt = x_i (f_i - mean_fitness)
```

This is the one-population or multi-population replicator equation in Skyrms §3. Each population has its own mean. Population definitions assign encounter slots to inherited types; a type can contain several slot plans. Independent individuals are drawn for separate slots, including slots assigned to the same population. Focal fitness averages those slots with explicit weights. Joint inherited-plan frequencies preserve correlations across inherited plans. Evolutionary rates and frozen populations are explicit configuration fields. The [domain-model guide](model.md) distinguishes this matching rule from the association of several functional modules within one participant.

`replicator_mutator.birth_uniform` implements Skyrms' mutation at reproduction:

```text
dx_i/dt = (1-epsilon) x_i f_i + epsilon mean_fitness/n - x_i mean_fitness
```

Mutation includes the parental type. This method requires nonnegative reproductive returns. Adding a common fitness background cancels in pure replicator selection; it changes the mutation term and reinforcement updates.

SciPy supplies numerical integration. Supported solvers are DOP853, RK45, and RK23. DOP853 is the default, with `rtol=1e-9`, `atol=1e-12`, and `max_step=1`. Implicit solvers require a future tangent-coordinate or analytic-Jacobian implementation because their componentwise finite-difference probes leave the simplex. Solver choice and tolerances belong to the numerical protocol. Invalid simplex excursions trigger smaller-step retries; probabilities are checked without clipping. Work, measurement, strategy, leaf, and compatibility-mask budgets bound the computation. Demonstrations initialize each population using a seeded Dirichlet distribution with concentration one; uniform initialization is available and can remain at a symmetry equilibrium.

Final-policy success and returns are exact expectations. Time-average measurements integrate expected success, returns, terms, and exchanges over evolutionary time. Stationarity residual, absent-type invasion advantage, and selection regret are separate diagnostics. A zero vector field alone does not establish evolutionary stability: every pure vertex is stationary under pure selection.

## Reinforcement

`reinforcement.conditional.R1` stores a propensity vector at each adaptive decision context. Each participating occurrence receives its addressed nonnegative return. Barrett adapters retain the encounter runtime's explicit feedback recipients and credit timing.

`reinforcement.strategy.R1` samples one complete role plan for each participant at the start of an encounter, executes its choices consistently, and adds that participant's nonnegative return to the chosen plan once. This provides the same role-plan representation used by the default separate-population evolutionary profiles. It is an explicit library learning counterpart; Skyrms' paper does not specify this historical simulation protocol.

Both methods preserve seeded generator state, weights, fixed components, and intervention histories. Evaluation leaves adaptive state and training randomness intact. Cumulative accuracy counts successes across training encounters; windowed accuracy uses the retained recent window; final-policy success integrates the frozen current policy over the game. `adaptive_ensemble` and `sg ensemble` retain independent seeds and method-specific horizons and metrics. Learning encounters and evolutionary time are distinct clocks.

`import_frozen_decisions` imports conditional responses from a checkpoint using explicit destination-to-source context bindings and optional option permutations. Its result supplies `fixed_decisions` and `lineage` for a new profile. Lineage retains the artifact hash, source configuration, seed, training clock, bindings, and policy transformation. Joint mixtures over complete plans need an explicit correlated representation when transferred.

## Skyrms game definitions

Source: Brian Skyrms, [Evolution of signalling systems with multiple senders and receivers](https://doi.org/10.1098/rstb.2008.0258), *Philosophical Transactions B* 364 (2009), 771–779.

| Identifier | Source setting | Default role-plan counts |
|---|---|---|
| `SK-SIG` | One sender, one receiver; two equiprobable states | 4, 4 |
| `SK-COMP` | Complementary partial observations; two senders, one receiver | 4, 4, 256 |
| `SK-VOTE` | Three independent noisy observations; receiver counts signals | 4, 4, 4, 16 |
| `SK-TEAM` | One sender addresses two receivers; four states require joint acts | 256, 4, 4 |
| `SK-CHAIN` | Sender, intermediary, receiver; two states | 4, 4, 4 |
| `SK-DIALOGUE` | A query selects the relevant observation partition | 64, 64 under each exposed encoding |

The complementary-information model preserves each sender's two-cell observational partition. Barrett's coding model gives its senders full state observations, yielding 16 sender plans each. The voting default has equiprobable states, independent observation error `0.1`, and anonymous signal counts. The resulting majority benchmark is `0.9**3 + 3*0.9**2*0.1 = 0.972`. Observation noise leaves a constant sender's signal constant; noise moved onto transmission is a distinct game.

Cost profiles explicitly select `prose_derived` or `table_literal`. Listening cost applies to a contingent complete receiver mapping. Conditional reinforcement with such plan-level costs is gated. Table-literal complete-plan reinforcement uses printed normal-form returns per encounter; prose-derived games use state-contingent returns. That return convention appears in fidelity metadata.

## Named source questions

| Question | Evidence and present treatment |
|---|---|
| SKQ1 — Historical numerical protocol | The paper refers to electronic supplementary simulations. Their implementation, initial sampling, solver, and stopping rules have not been recovered. Historical reproduction profiles remain gated; demonstrations record their own protocols. |
| SKQ2 — Cost-table arithmetic | Tables 2–3 print `1.5` for constant senders against contingent receivers, although the stated biased state probabilities imply `4/3` or `5/3`. Both printed and prose-derived interpretations are exposed; printed `1.33` and `1.67` remain literal. |
| SKQ3 — Learning with listening costs | The paper supplies a cost for attending to messages but no conditional-urn credit rule for assessing a whole mapping. Whole-plan reinforcement supports the mapping-level payoff; conditional use remains gated. |
| SKQ4 — Dialogue reply information set | The observer's reply may bind to the chosen partition or the incoming query. `partition_cell` and `query_cell` expose these encodings explicitly; their off-equilibrium strategies differ. Four-state probabilities are explicit. Dialogue paper-game defaults remain gated. |
| SKQ5 — Grouped inheritance details | The paper permits one-, two-, and three-population models. Grouped profiles declare composite inherited plans, independent role matching, and role weights. These operational choices are retained in the population specification. |
| SKQ6 — Reversed voting example | The prose example pairs reversed signals with an inconsistent act label. The implementation identifies the state from majority observations and tests both signal conventions against the same payoff rule. |

## Exact comparisons

The dyadic signalling game supplies the simplest shared comparison: four complete sender plans and four receiver plans, the same state distribution, and the same success matrix. An evolutionary distribution and normalized complete-plan learning propensities can be evaluated by the identical expectation function. Complete-plan distributions can contain correlations between a role's conditional choices; replacing them by independent context marginals changes the policy.

The frozen translator in §5 gives an analytically tractable one-role control. With the source's sender and receiver held fixed, the four translator plans have returns `[0.5, 0, 1, 0.5]` in the library's lexicographic order. Pure replicator frequencies have the closed form `x_i(t) = x_i(0)*exp(f_i*t) / sum_j(x_j(0)*exp(f_j*t))`. The unique optimal translator has return one. Both learning representations can be evaluated exactly against these endpoints.

Complementary information, voting, teamwork, and chains also permit exact final-policy evaluation under matched game parameters. A controlled comparison records representation, initial policies, seeds, frozen components, and budgets, and reports final-policy success and participant returns. Training curves retain their native clocks. Numerical evolution and sampled reinforcement produce the policies; exhaustive expectation makes their evaluation exact up to floating-point arithmetic.
