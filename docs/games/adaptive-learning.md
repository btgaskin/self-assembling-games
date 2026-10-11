# Adaptive learning and attention

## BOX — Fixed and random box tasks

**Source:** [B11 §11.1, pp.188–191](https://doi.org/10.1093/9780197793572.003.0011).

Two opaque boxes are possible actions. In the fixed task, choose the rewarded box initially at random and replenish that same box after success. A simple reinforcement learner has one A/B urn initially one each; reward successful choices by one. A win-stay/lose-shift comparator keeps its action after success and switches after failure.

In the random-box task, select a special box uniformly once per run. On every play, independently put food in that special box with probability `2/3` and the other with probability `1/3`, regardless of the previous outcome. The reported ten-play fixed-box comparison favors WSLS (.95 versus .74 for simple reinforcement). For one-thousand-play random-box runs, simple reinforcement does better (.63 versus .55 for WSLS), using one thousand runs in the source. The task type and horizon therefore matter to the meta-learning example.

## CHANGE — Adapting a code after payoff change

**Source:** [B11 §11.2.5, pp.206–211](https://doi.org/10.1093/9780197793572.003.0011), Barrett (2014).

Start with COD, train under the identity state→matching-act reward rule, then swap the successful actions associated with two states. Carry the learned vocabulary and urns into the new phase. Bound prior reinforcement or forget old experience so the policies can change. A new code can arise through changed receiver interpretations or changed coordinated sender partitions.

The book describes bounded reinforcement and reports an illustrative .58/.42 split between receiver-side and sender-side adaptation. A canonical numeric bounding mechanism and its parameters need the cited paper. Each option-count ceiling, total-urn ceiling, or discount rule must have its own configuration label. Preserve phase length and intervention time, and analyze term-wise partitions separately from whole-expression correctness.

## INV — Inventing and forgetting signals

**Source:** [B11 §11.2.6, pp.211–216](https://doi.org/10.1093/9780197793572.003.0011), Alexander, Skyrms and Zabell (2012).

Use basic signaling with fixed finite state and act domains but an expandable signal alphabet. Sender urns begin with one mutator; receiver urns begin absent. R5 governs first-use acceptance and rollback. Receiver rows contain one of each act. Failure of an existing signal causes no addition; success reinforces sender and receiver by one.

The chapter reports high signaling efficiency in uniform three-state and strongly biased two-state games. The signal alphabet has no fixed finite maximum in this model; a resource guard must be reported as an engineering limit, and reaching it recorded.

The forgetting extensions select either a random ball or a random urn followed by a random represented type and remove a ball. The latter can reduce rare synonyms more effectively because it samples types rather than their frequency. Specify forgetting frequency, treatment of the mutator, extinction of ordinary types, and cleanup of receiver rows before enabling it. The source also discusses mutation/crossover of structured actions and externally correlated strategies; those are additional mechanisms, not implicit properties of an invented string ID.

## HYB — Persistence with reward memory

**Source:** [B11 §11.3.2, pp.220–223](https://doi.org/10.1093/9780197793572.003.0011).

Apply R6 to SIG. Keep the current conditional signal and action assignments as well as the reward urns. Success retains assignments and strengthens their recorded weights; failure changes only the currently visited assignments through a weighted resample. Human-subject models discussed nearby include inertia and mixtures of concurrent learning processes; their extra parameters define distinct learner profiles.

## BAND — Reinforced choice among bandits

**Source:** [B12 §12.2, pp.239–242](https://doi.org/10.1093/9780197793572.003.0012).

One urn holds three machine choices A,B,C, initially one each. Choose a machine, observe a payoff between one and ten, then add that payoff to its weight. Reward distributions are fixed within the task but unknown to the learner. The book's argument relies on an arm whose expected reward dominates the alternatives by a factor greater than one under specified reward conditions. The actual machine distributions need to be supplied; a convenient three-Bernoulli demonstration is a separately labeled variation.

## META-RULE — Reinforcing learning-rule selection

**Source:** [B12 §12.3, pp.243–247](https://doi.org/10.1093/9780197793572.003.0012).

An outer urn chooses among simple reinforcement, win-stay/lose-randomize, and reinforcement with iterative punishment. A selected rule runs a **fresh** inner game with its own initialized state. The task profile names game size, inner duration, and success criterion, such as cumulative accuracy above a threshold. A successful outer choice gains one; an unsuccessful choice gains a specified small positive amount. The outer urn persists across trials.

A second illustration chooses between WSLS and simple reinforcement on repeated ten-play fixed-box tasks. Each task resets box location and inner learner state. Reinforce the outer rule by its achieved cumulative accuracy. The chapter considers many repeated problems, including ten thousand and one million. The relevant accumulated experience belongs to the meta learner; carrying the same inner box urn across all problems would change the task.

## META-ORACLE — Reinforcing adviser choice

**Source:** [B12 §12.3, pp.245–247](https://doi.org/10.1093/9780197793572.003.0012).

One urn chooses among seven oracles/advisers, initially one each. Reinforce the chosen oracle by one for a correct answer and by a specified small positive amount for an incorrect answer. The question stream and each oracle's answer process define the environment. Reliability need not be modeled as independent Bernoulli trials unless the selected source profile makes that assumption. This is the same outer-choice mechanism with adviser outputs in place of lower-level learner results.

## MH — Monty Hall as a learning problem

**Source:** [B11 §11.4, pp.226–232](https://doi.org/10.1093/9780197793572.003.0011); [B12 §12.3](https://doi.org/10.1093/9780197793572.003.0012).

The standard three-door environment requires a prize-placement rule, initial choice, and a host who opens an eligible unchosen goat door. A reinforcement decision between staying and switching can learn from final wins; the host policy is essential to the expected payoffs. The chapter's Bayesian account is a separate dynamics. Catalogue both interpretations, and use the specified host protocol when making an RL demonstration rather than treating any door-opening process as equivalent.

## GEN / SET — Generalization and learning sets

**Source:** [B12 §§12.4–12.5, pp.247–259](https://doi.org/10.1093/9780197793572.003.0012), O'Connor (2017); Harlow's experimental learning sets; Torsell and Barrett (2024).

Generalization spreads the effects of experience across similar circumstances. The similarity relation, reward distribution over neighbors, and context representation define a usable model. The chapter discusses the dependence of useful generalization on the task rather than choosing one universal kernel.

Learning sets concern improved performance across repeated problems. The Harlow experiments are empirical examples, with object identities and positions separating relevant from irrelevant features. The book discusses progressively stronger reinforcement/punishment as a way to approximate WSLS; a numerical schedule remains to be pinned. Population evolution of learning in Smead and O'Connor belongs to a later population-dynamics family.

## ATT-R — Receiver attention

**Source:** [B12 §12.6, pp.259–264, Fig.12.2](https://doi.org/10.1093/9780197793572.003.0012), Herrmann and VanDrunen (2025).

Nature chooses A/B uniformly. The sender has two state urns with options 0/1, initially one each; it sets the world's signaling feature `f0` to the drawn value. The receiver may observe any of N binary features, including `f0`. The other features have randomly determined values; a profile must pin their sampling and correlations.

The receiver has one salience urn over the N feature identities, initially one each, and two action urns per feature, keyed by its value, each containing A/B once. It draws a feature, observes its value, draws an act from that feature/value row, and acts. A correct act reinforces the sender's signal choice, receiver's salience choice, and receiver's action choice by one. Failure pays zero.

There are `3+2N` urns and `4+5N` weights, derived from the specification. For `N=1` with only `f0`, this reduces to the ordinary two-state game. If `f0` is absent and distractors carry no state information, reliable signaling through the sender cannot emerge. The book reports mean cumulative accuracy over .90 after one million plays with one spurious feature and over .80 with four.

## ATT-B — Sender and receiver attention

**Source:** [B12 §12.6, p.263](https://doi.org/10.1093/9780197793572.003.0012).

Add a sender salience urn choosing between the true task-relevant source and binary spurious sources. The sender observes the selected source/value. The binding from that observation to conditional signal urns is a source-profile decision: source/value-specific tables and shared value-only tables are explicit alternatives. The receiver retains ATT-R's feature selection. Reward all participating salience and action decisions on a match to the true task state.

The chapter reports reliable signaling above .75 in about .68 of runs with one true and one spurious source on each side, and about .14 with four spurious sources on each side. It specifies the extension's mechanism; the exact conditional-table sharing and distractor law should be checked in Herrmann and VanDrunen before a canonical reproduction. This distinction matters because sharing a signal urn across sources constrains the learner differently.

## RIT — Ritualization of learning itself

**Source:** [B12 §12.7, pp.264–270](https://doi.org/10.1093/9780197793572.003.0012).

The general account allows saliences, actions, and the dispositions that update them to sharpen together. ATT-R and ATT-B instantiate concrete portions of that account with prespecified urn learners and feature alternatives. A fully unconstrained ritualization model needs a specified primitive adaptive process and emergence criterion. Preserve this as an explanatory proposal rather than supplying an arbitrary “all structures emerge” algorithm.
