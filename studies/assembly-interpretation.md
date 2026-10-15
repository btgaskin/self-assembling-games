# What the first assembly comparisons show

The extension makes acquired organisation available as an experimental object:
we can retain it, release a constraint, copy it into a new task, and compare the
result with a declared control. The [endpoint report](assembly-smoke.md) contains
129 trials across 43 conditions, with three seeds per condition. These are short
exploratory runs; the longer lifecycle proposals remain in `pilot.toml` files.

## Acquisition and the initial structure

CUE and MAN release both reached mean success about 0.973 after 100 precursor
encounters and 100 released encounters. Uniform fresh signalling reached 0.847
at 100 encounters and 0.919 at 200. That comparison initially looked like a large
benefit from acquired organisation. It also bundled in the declared 9:1 release
bias, so a final control released the fixed side after zero precursor training.
This control reached 0.953 for CUE and 0.960 for MAN. Acquired history contributed
a smaller improvement once that supplied initial structure was represented.

This is a useful design lesson for the kit: a lifecycle intervention is itself
part of the apparatus. Its numerical parameters belong in the configuration, and
its contribution deserves a matched control. The almost symmetric CUE/MAN results
also make their symmetric two-state observation and payoff structure apparent.

## Reuse depends on the acquired component

Longer source training improved NAND-to-OR transfer from 0.594 to 0.719 mean
success, and one-selector composition from 0.555 to 0.683. Perfect supplied NAND
components reached 0.828 in transfer and 0.884 in one-selector composition.
These comparisons expose the capacity that a perfect fixture supplies: learning
an interface around an imperfect component is a different problem.

At these budgets, learned transfer remained below direct OR learning at the same
online budget (0.734), and below direct learning at matched total exposure
(0.765). The experiment therefore demonstrates usable acquired-state transfer
without yet showing an efficiency advantage. Component accuracy, acquired coding
conventions, translator difficulty and total acquisition cost should remain
separate measurements in the next pilot.

The one-selector adaptive route reached 0.683, against 0.675 with its uniform
route frozen. This small early difference leaves the value of routing adaptation
open at longer horizons. The two-selector acquired composition reached 0.623;
its larger adaptive problem deserves its own learning curve. Internal NAND
components remain frozen in these studies. A coadaptive-internals extension
would require an explicit rule for crediting their participating decisions.

## Joint learning, similarity and conflict

The three LaCroix conditions reached mean success between 0.192 and 0.235, with
substantial variation across seeds. At 100 encounters they provide an exercised
joint-learning mechanism and early dispositions. Longer learning curves are
needed to compare the eventual effect of sender information and relevance rules.
The implementation checks symbolic pair correctness, efficient participation and
context interpretation separately from Boolean output coincidence.

Binary generalization illustrates a stronger structural result. Mean expected
reward declined from 1.886 at alpha=0 to 1.713 at alpha=0.1; alpha=1 gave exactly
1.5 in every seed. Gaussian width 2 increased mean reward from 0.255 at width 0
to 0.309, while exact-match accuracy remained low. A wider kernel reinforces
nearby contexts and also adds more total reinforcement in this unnormalised
protocol. A matched-total-reinforcement variant would separate those effects.

In the constructed SCOD conflict grid, mean task success declined monotonically
across the five tested mixture values, from 0.345 at lambda=0 to 0.254 at lambda=1.
The retained participant utilities show whose objective each update serves.
The published signalling comparators exercise a second distinction: informative
behaviour, participant utility and a task-correctness predicate can move apart.
Their report includes exact state–signal–action distributions and mutual
information. One unit of replicator time and 100 reinforcement encounters are
separate early endpoints, so their differing scores do not establish that one
adaptive mechanism learns faster.

## A compact mathematical description

For the generalized-reinforcement case, write the propensity table as a matrix
Q with contexts in rows and actions in columns. A rewarded action a in context s
updates it by

```math
Q_{t+1} = Q_t + r_t K[:,s] e_a^T.
```

For each action, the change from its initial context vector lies in the column
space of K. Thus the reachable changes occupy a subspace of dimension at most
(number of actions) × rank(K). The actual nonnegative updates further restrict
which points in that subspace can be reached.

For the binary kernel, K=[[1,alpha],[alpha,1]]. At alpha=1 its rank is one and
identically initialised rows remain identical. With uniform states, the action
becomes independent of the state: exact-match probability is 1/2 and expected
reward is 1.5. This is a direct consequence of the update rule, and the tests and
runs check it exactly. It provides a useful mathematical connection between an
exposed material coupling and a limit on what a learner can acquire. It is a
reformulation of this mechanism, with no claim of mathematical novelty.

## Next experiments

Run the retained lifecycle pilots first, extending the seed count before making
comparative claims. Preserve matched release priors, component competence at
export, online versus total exposure, and endpoint policy evaluations. Add
normalised generalization kernels as explicit variants and inspect both reward
and discrimination. For conflict, follow utilities, information and stationarity
across longer native-clock trajectories. Joint composition should compare frozen
reuse with a separately specified internal-credit rule before it is described
as coadaptive component learning.

The foundation checks support these studies: negative payoffs retain their
intended domain, zero-reproduction censuses remain restorable, portable components
retain their state without original files, and phase receipts bind every result
to its declared seed, parent and trial inventory. The full suite passed 955 tests.
