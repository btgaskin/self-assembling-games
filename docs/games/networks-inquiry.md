# Networks and inquiry

## VIS — Reinforced visiting networks

**Source:** [B9 §9.2, pp.150–152](https://doi.org/10.1093/9780197793572.003.0009), Skyrms and Pemantle (2000).

Each of N agents has a visiting urn containing each of the other N−1 agents at weight one. Self-visits are excluded. During a round every agent chooses one host and plays the specified game. Reinforce the visitor's chosen host by the visitor's payoff. In a reciprocal version, also reinforce the host's visiting option for that visitor by the host's payoff. A game can supply constant reinforcement to study network formation independently of strategic adaptation.

This is a weighted directed network: source is the visiting agent, destination is its chosen host, and normalized propensity is visit probability. A reciprocal learning rule can create reciprocal ties without forcing symmetric weights. The schedule of visits and whether strategy updates occur immediately or at a round boundary belong to the configuration.

## STAG — Fixed-strategy stag/hare network

**Source:** [B9 §9.3, pp.152–155](https://doi.org/10.1093/9780197793572.003.0009).

Assign each agent a fixed stag or hare strategy. Use VIS with reinforcement for both visitor and host. In the book's payoff scale:

| Row player / partner | Stag | Hare |
|---|---|---|
| Stag | 4 | 0 |
| Hare | 3 | 3 |

Fixed types can learn to select useful partners. This assembles the interaction network while holding strategies constant. The cited original uses an equivalent payoff ordering on another scale; absolute reinforcement scale affects convergence speed relative to initial weight and needs its own profile.

The chapter also discusses rare imitation of the best-performing strategy from the preceding round. That combines reinforcement over visits with a different strategy update. Its imitation probability and payoff accounting must be specified before reproducing the coevolution example.

## FFSR — Signaling conventions and partner choice

**Source:** [B9 §9.4, pp.155–157](https://doi.org/10.1093/9780197793572.003.0009), Fulker, Forber, Smead and Riedl (2024).

Each agent has a visiting urn over other agents and a strategy urn over sixteen complete plans for a two-state/two-signal/two-action game. A plan combines one of four deterministic state→signal maps with one of four deterministic signal→act maps. Every plan initially has equal propensity.

On a visit, both agents sample their plans. Calculate the **expected** common success over uniform nature and an equal chance of either player occupying the sender role. That expected payoff reinforces both agents' selected strategies and the visitor's selected host. The host's visiting urn is unchanged. This differs from selecting a single nature state and reinforcing on one realized match.

The model introduces forgetting/error mechanisms to retain exploration. The book establishes the interaction and update recipients; exact rates and the source's decay/error order require the original paper. Keep this as a source-clarification case rather than inventing a default rate and calling it canonical.

## SUB — Subscription networks and information value

**Source:** [B9 §9.5, pp.157–164](https://doi.org/10.1093/9780197793572.003.0009), Bala and Goyal (2000).

Agents can pay for information links, and benefits depend on reachable information through the network. This requires a set of outgoing subscriptions, a reachability calculation, link costs, and unilateral link-change opportunities. The chapter contrasts unsuccessful generic reinforcement approaches with best-response learning and inertia. It does not choose a complete numerical urn implementation for the subscription example. Catalogue the model and its state now; defer a canonical reinforcement runner until its cited rule is recovered.

## DISC — Discrimination in evolving bargaining networks

**Source:** [B9 §9.6, pp.164–167](https://doi.org/10.1093/9780197793572.003.0009), Rubin and O'Connor (2018).

The cited bargaining/network example uses adaptive partner choice and strategic responses to explain persistent group-level inequity. Its rewiring is a best-response mechanism rather than the ordinary reinforcement visiting rule. It is a related network model with a separate update contract, included in the catalogue for coverage of the chapter.

## INQ — Epistemic-network inquiry

**Source:** [B10 §10.1 and following comparisons, pp.170–185](https://doi.org/10.1093/9780197793572.003.0010).

Each of N inquirers receives fixed epistemic reliability `e_i` and social reliability `s_i`. The illustrative setting samples these independently and uniformly on `[0,1]`. Each inquirer has one consultation urn with N+1 options: nature and **all N inquirers, including itself**, each initially one.

A fresh inquiry problem starts everyone with a false belief. Urn weights survive across problems. Each round orders the inquirers by a fresh random permutation. Each inquirer draws a source and updates its belief immediately:

- Nature: obtain a true belief with probability `e_i`, otherwise a false belief. This is a new empirical consultation each time.
- Another inquirer j: obtain a true belief only if j currently has a true belief and a Bernoulli draw with probability `s_i` succeeds. Social reliability belongs to the **consulting** inquirer i. Otherwise its new belief is false.
- Itself: retain the current belief. A currently true belief stays true through self-consultation; this does not apply an additional social-error trial.

If the consultation produces a true belief, add one to the chosen source in the consulting inquirer's urn; otherwise return without addition. Consulted sources receive no reward. This externally supplied success feedback is an assumption of the model, even where an inquirer does not explicitly know which of its beliefs are true.

Run the prescribed number of rounds, then reset all beliefs for the next problem while retaining learned consultation propensities. The chapter uses two hundred empirical generations of one hundred complete rounds in an example and studies groups including N=5 and N=20. An empirical generation is this reset-and-repeat unit.

**Variants:** set all social reliabilities to one for perfect communication; vary epistemic or social reliabilities to study how reliable empirical sources and communication interact with network structure. Keep parameter draws fixed within a run and distinguish comparisons that redraw agents from interventions on an existing group.

Measure beliefs over time, selection of nature, self-consultation, influence through directed consultation edges, and final source probabilities. A graph display can threshold tiny probabilities for readability; retain the full weighted network in results. Every initially available edge remains represented under positive reinforcement.

Inquiry can evolve reliance on particular empirical investigators and internal control over information flow. Its asynchronous beliefs, self-option, and cross-problem persistence are all part of the mechanism. A synchronous imitation process with no nature node would define a different game.
