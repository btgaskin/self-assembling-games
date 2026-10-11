# Learning rules

The equations below specify the core updates. A game supplies observations, eligible decisions, and payoff. Its profile supplies initial values and any additional schedule. Source terminology is retained; in behavioral psychology, punishment and negative reinforcement have distinct meanings even though Barrett sometimes uses “negative reinforcement” for subtracting propensity.

## R1 — Simple reinforcement

For the chosen option `k` in the visited urn:

`q_i' = q_i + π · 1[i=k]`, with `π ≥ 0`.

The response rule is `p_i = q_i / Σ_j q_j`. Standard signaling initializes each option at one and uses payoff one for success, zero for failure. Unvisited urns remain unchanged. Positive initial values act as initial propensity in addition to accumulated rewards. Fractional positive payoffs work directly. [B2 §2.2, pp.23–25](https://doi.org/10.1093/9780197793572.003.0002).

Small initial weights relative to rewards give early outcomes more influence. Large initial weights slow changes in probabilities. Unequal initial weights bias the starting policy. Record this ratio rather than treating “one ball” as a scale-free choice.

## R2 — Reinforcement with punishment

For the chosen option, success adds `a`, failure subtracts `b`, with a floor of one:

`q_k' = q_k + a` on success; `q_k' = max(1, q_k − b)` on failure.

Other weights remain unchanged. Barrett writes this as `(+a, −b)`. The chapter 2 example uses `a=b=1`; later examples vary both. The floor keeps every existing option available. This rule addresses pooling but its effectiveness depends on reward, punishment, and task complexity. [B2 §2.4, pp.35–36](https://doi.org/10.1093/9780197793572.003.0002); [B11 §11.2.2, pp.195–199](https://doi.org/10.1093/9780197793572.003.0011).

For signed net payoffs in the hierarchical game, use `q_k'=max(1,q_k+π)` where `π=base_success_reward−total_signal_cost`. This also permits punishment after a successful action whose cost exceeds its base reward. [B3 §3.3, pp.49–50](https://doi.org/10.1093/9780197793572.003.0003).

## R3 — Reinforcement with iterative punishment

Barrett writes `[+a, −b]` for alternating phases:

- Reinforcement phase: add `a` on success; leave weights unchanged on failure.
- Punishment phase: subtract `b` on failure with the chosen-option floor; leave weights unchanged on success.

The experiment specifies phase lengths and number of cycles. In the 32-state example, `[+2,−4]` runs for one million plays. Two iterations mean four equal quarters: reward, punish, reward, punish. Four, eight, and sixteen iterations divide that same total into eight, sixteen, and thirty-two phases. A clock controls the phase, rather than separately drawing which rule to use on each play. Self-tuning phase lengths are discussed as an extension and require an explicit schedule or criterion. [B11 §11.2.3, pp.199–202](https://doi.org/10.1093/9780197793572.003.0011).

## R4 — Adjustable-reference-point reinforcement (ARP)

Using chosen option `k`, payoff `π_k`, and the old reference point `ρ`, define `R=π_k−ρ`:

`q_i' = max(v, (1−φ)q_i + E_k(i,R))`.

The experience function printed in the book is

`E_k(i,R) = R(1−ε)` if `i=k`, and `Rε` otherwise.

Update the reference point with `ρ'=(1−w)ρ+wπ_k`, where `w=w+` if `π_k≥ρ`, otherwise `w=w−`. Normalize the new propensities with the ordinary proportional response rule. Use old `ρ` for both comparisons and the reward calculation; then replace it.

Parameters in the chapter's footnote are `ε=.2`, `v=.0001`, `φ=.001`, initial `ρ=0`, `w+=.01`, `w−=.02`; initial propensities are a further specified parameter. This update affects every option in the relevant learner's propensity vector. The scope of reference points across multiple conditional urns needs a source decision before using it as a canonical act-based game learner.

The printed unchosen increment is **per unchosen option**, with no division by the number of alternatives. For more than two options, its total differs from a normalized redistribution of `Rε`. Keep this literal transcription visible; check Bereby-Meyer and Erev (1998) before establishing a corrected profile. The book also alternates ARP/APR spelling and payoff indices. [B11 §11.2.4, pp.202–206, especially pp.204–206](https://doi.org/10.1093/9780197793572.003.0011).

## R5 — Reinforcement with invention

Initial sender urns contain one mutator option each; receiver urns are created as needed. Drawing the mutator proposes a fresh signal token. A receiver encountering it creates a provisional urn with one ball of each possible act.

On first-use success, add the new signal at weight one to **every urn belonging to that sender**, then reinforce the chosen sender context and receiver action. Keep the receiver urn. On first-use failure, discard the new signal and its provisional receiver urn. Existing signals use R1.

A pair-sender game needs a transaction containing both proposed signals and the dependent pair urn. Decide its commit and rollback according to that game's source. Signal IDs should remain unique even after rollback to make traces unambiguous; this ID policy is an engineering choice.

The Alexander–Skyrms–Zabell version is explicitly described in [B11 §11.2.6, pp.211–216](https://doi.org/10.1093/9780197793572.003.0011). Bird ordering and addition use related invention mechanics, specified in their own game sheets. Their receiver-urn retention on failed inventions should be verified separately rather than assumed from the single-sender case.

Forgetting variants include removing a randomly selected ball or first selecting an urn and then a represented type uniformly. These are different probability measures. Whether mutators are exempt and whether the last ball of an ordinary type can disappear requires the cited original implementation.

## R6 — Win-stay/lose-randomize with reinforcement

Keep a current deterministic conditional choice alongside each urn. Initially choose current assignments uniformly and initialize counts at one. A successful play keeps the current assignments and adds one to the used sender and receiver options. After failure, resample only the current sender choice for the state just seen and the current receiver choice for the signal just received, using their accumulated urn weights. Other assignments persist; a resample can return the same option. [B11 §11.3.2, pp.220–223](https://doi.org/10.1093/9780197793572.003.0011).

This combines an act-based persistence rule with reward memory. Its current assignment is additional state, so implementing it as ordinary proportional sampling on every play changes the dynamics.

## R7 — Outer reinforcement over rules or advisers

Treat a learning rule or adviser as an urn option. Draw it, run the lower-level problem or consultation, and reinforce that option using the specified outer return. Keep outer weights across fresh inner tasks. Returns can be cumulative accuracy, a success indicator, or success plus a small positive payoff on failure. Record each variant's return function and reset boundary. [B12 §§12.2–12.3, pp.239–247](https://doi.org/10.1093/9780197793572.003.0012).

## Additional configurable mechanisms

Forgetting discounts old propensities; bounded reinforcement limits accumulated commitment. Barrett discusses both without choosing one universal bounding algorithm. A per-type ceiling, an urn-total bound, and multiplicative decay produce different response trajectories. Each needs its own named profile. Generalization reinforces related options according to similarity; the similarity relation and spread of reward are part of that model. [B11 §§11.2.1,11.2.5, pp.193–195,206–211](https://doi.org/10.1093/9780197793572.003.0011); [B12 §12.4, pp.247–252](https://doi.org/10.1093/9780197793572.003.0012).

## Comparators and later dynamics

Strategy-based win-stay/lose-randomize in B2 replaces both **whole maps** on a failed play, choosing them randomly; success keeps both maps. It eventually finds an absorbing perfect convention in the stipulated finite, solvable, noise-free signaling games with recurrent states, but the search can be long. It is useful as a comparator and is distinct from R6. Win-stay/lose-shift, probe-and-adjust, best-response/inertia network updating, Bayesian conditioning, and population replicator dynamics have their own state and response contracts. [B2 §2.4, pp.34–35](https://doi.org/10.1093/9780197793572.003.0002); [B9 §§9.5–9.6](https://doi.org/10.1093/9780197793572.003.0009); [B11 §§11.3–11.4](https://doi.org/10.1093/9780197793572.003.0011).

## Replicator connection

For a fixed urn with large total weight `Q`, conditional expected reward `u_i`, and small reinforcement, the leading expected probability increment is proportional to `p_i(u_i−Σ_jp_ju_j)/Q`. This explains the relation to a replicator-style mean-field vector field. Conditional visitation rates and growing urn totals affect time scaling; stochastic sample paths retain their own dynamics. Barrett cites Beggs (2005), Hopkins and Posch (2005), and Börgers and Sarin (1997), distinguishing simple reinforcement from Bush–Mosteller learning. A future deterministic backend should derive its field from the chosen game and learner, with an explicit time convention. [B2, pp.25–27](https://doi.org/10.1093/9780197793572.003.0002).
