# Self-assembling games

A research library for **self-assembling games**: organized interactions that acquire signaling conventions, computation, coordination, and inquiry through adaptive decisions. Game definitions describe what participants observe, choose, and receive. Adaptive methods describe how their decisions or inherited strategies change.

The executable Python package separates games, method classes, and experimental protocols. Its common unit is a **conditional decision**: a component observes a circumstance and selects an option. Reinforcement changes its response through feedback; population evolution changes the frequencies of complete inherited plans. Connecting decisions produces signaling, translation, attention, questions, partner selection, and learning-rule selection.

The source families currently include Jeffrey A. Barrett's *Self-Assembling Games: Language, Learning, and Inquiry* (2025), with **39 catalogue entries, 32 executable model families, and 45 encounter profiles**, and Brian Skyrms' *Evolution of signalling systems with multiple senders and receivers* (2009), with **six finite game families**. There are also **40 adaptive profiles** spanning the Skyrms games and selected Barrett adapters. Each profile records its game, method, assumptions, and protocol.

Source-specific definitions preserve observations, option sets, feedback, and timing. The finite core supports continuous-time replicator dynamics, birth-mutation replicator dynamics, conditional reinforcement, and complete-plan reinforcement. The Barrett encounter runtime additionally supports punishment, persistence, invention, and nested learning. Source questions and deliberate variants travel with configurations and checkpoints. See the [shared adaptive architecture](docs/finite-dynamics.md) and [source register](docs/sources.md).

## Games, methods, and history

The library takes self-assembling games as its organizing concept and traces each model's lineage through its source family. Lewis' sender–receiver game provides the convention-forming setting. Skyrms' 2009 paper studies evolutionary dynamics and extends that setting to complementary information, voting, teamwork, chains, and dialogue. Barrett develops conditional learning, modular composition, and the assembly of interaction and inquiry. The [chapter source register](docs/sources.md) records the antecedents of individual Barrett models.

| Method class | Finite method | Adaptive state |
|---|---|---|
| Population evolution | `replicator.continuous` — default | Frequencies of complete inherited plans |
| Population evolution | `replicator_mutator.birth_uniform` | Plan frequencies with uniform mutation at reproduction |
| Reinforcement | `reinforcement.conditional.R1` | Propensities at permitted decision contexts |
| Reinforcement | `reinforcement.strategy.R1` | Propensities over complete role plans |

Solvers are separate from mathematical methods. The default replicator equation is Skyrms' continuous-time equation; DOP853 is the library's default numerical solver. Population grouping, role weights, fixed components, initialization, and mutation are explicit choices. Full inherited distributions retain correlations between roles. Additional finite games can be registered through `register_finite_game`.

Install evolutionary support and list the families:

```sh
python -m pip install -e '.[dev,evolution]'
sg list --all
sg list --family skyrms
sg methods
sg run sk-sig.evolution.demo --seed 1 --time 20 --output runs/signal-evolution
sg run sk-sig.strategy-r1.demo --seed 1 --plays 10000 --output runs/signal-learning
sg evaluate runs/signal-evolution
sg evaluate runs/signal-learning
```

These two signalling profiles use the same finite game and complete role-plan sets. Their final policies can be evaluated exactly by summing over all states and strategy combinations. Evolutionary time and learning encounters are reported separately. The frozen-endpoint translator offers a particularly simple comparison with one adaptive role and known plan payoffs; the voting game supplies an exact majority-rule success benchmark of `0.972`. See [comparison conditions and source questions](docs/finite-dynamics.md).

The Skyrms profiles specify demonstration protocols. Historical reproduction remains gated by the missing supplementary simulation protocol, and dialogue exposes its unresolved reply information set as a named configuration choice.

## The book and its central question

Barrett asks how agents acquire the dispositions that make an organized interaction possible. A sender and receiver can begin with arbitrary signals and random responses. Success reinforces the decisions that happened to work together. Over repeated encounters, they may acquire a convention: a signal comes to indicate a state and elicit a useful action.

This is the first level of self-assembly: conditional acts become complete strategies. Subsequent games let agents learn what to attend to, which components to activate, how to translate new inputs into an old representation, when to ask a question, and whom to consult. These choices assemble the interaction itself. Existing games can become parts of a larger game, while learned connections determine how those parts are used.

The explanatory aim is to show how such capacities can arise under specified, modest learning resources. Signals acquire meaning through their information and their role in successful action. Computation and inquiry emerge through related forms of coordination. The book's conclusion develops these as **how-possible explanations**: the mechanism shows a way the capacity could arise under the model's conditions. See [B1 and B13 in the source register](docs/sources.md).

| Chapter | Subject | What it contributes |
|---|---|---|
| B1 — Self-Assembly and Games | Decisions and adaptive dispositions | The explanatory framework for assembling games |
| B2 — Learning to Signal | Lewis–Skyrms signaling and reinforcement | Conditional urns, learned conventions, pooling, and alternative learning rules |
| B3 — Generalized Signaling Games | Coding, NAND, and hierarchy | Joint expressions, logical computation, and executive control |
| B4 — Assembling Games from Decisions | Cue reading and sensory manipulation | One-sided precursors and the ritualization of interactions |
| B5 — Rule Following and Template Transfer | Ordering and logical templates | Applying an acquired structure to new stimuli and incomplete training |
| B6 — Modular Composition and the Evolution of Salience | Chains and source selection | Learning the connections that join components |
| B7 — The Self-Assembly of Computation | NAND networks and arithmetic | Composing acquired computational capacities |
| B8 — Learning to Take Turns | Questions and dialogue | Learning when to request information, reply, and act |
| B9 — The Evolution of Networks | Visiting, subscriptions, and bargaining | Learning interaction partners and coevolving conventions |
| B10 — The Self-Assembly of Inquiry | Empirical and social consultation | Learning information sources and organizing epistemic networks |
| B11 — The Ways of Learning | Punishment, reference points, invention, and persistence | Comparing adaptive dynamics on different learning problems |
| B12 — Learning How to Learn | Rule selection, generalization, and attention | Adapting how experience is acquired and used |
| B13 — Self-Assembling Games and Explanation | Explanatory synthesis | Interpreting the capacities and structures that the games produce |

References use **B1–B13**, section numbers, and printed pages. The [source register](docs/sources.md) supplies chapter DOI links, quotation anchors, and Barrett's cited antecedents. The [chapter manifest](sources/chapters.json) records Zotero identifiers and attachment fingerprints.

## Components and capacities

An **agent** is an organism or social participant. A **module** is a functional component belonging to an agent. Its **role** describes what it does in an interaction: sender, receiver, translator, executive, source selector, or learner selector. One animal can contain several modules. A capacity can arise from several modules working together.

The following vocabulary organizes the implementation. It groups mechanisms described across the book; it is a repository design rather than a formal taxonomy attributed to Barrett.

| Component or capacity | What it observes and chooses | How it is represented |
|---|---|---|
| Conditional action | A permitted observation determines a choice among acts | A context-bound propensity table, or **urn** |
| Signaling | A sender sees a state or feature and chooses a signal | Sender tables; initially arbitrary signal identities |
| Interpretation | A receiver sees a signal or ordered expression and chooses an act | Receiver tables keyed by the received expression |
| Coding and composition | Separate signals distinguish complementary aspects of a state | Sender-specific terms and a joint receiver context |
| Salience and attention | A component chooses which available source or feature to observe | A source-selection urn followed by the selected observation |
| Executive control | Context determines which senders act or how an output is interpreted | Activation and interpretation decisions with explicit costs |
| Translation and template use | A new stimulus selects an input of an acquired game | Adaptive translator tables connected to frozen old modules |
| Computational composition | One module's output supplies another module's input | Explicit connections and representation adapters; shared outputs are reused |
| Invention | A mutator proposes an additional signal type | Provisional tokens, new receiver rows, acceptance and rollback rules |
| Dialogue and turn-taking | A participant chooses to ask, reply, continue, or act | Sequential decisions, occurrence history, and termination rules |
| Partner selection | A visitor chooses another agent for an interaction | Visiting urns whose probabilities define a directed network |
| Inquiry | An inquirer chooses nature, itself, or another inquirer | Consultation urns, current beliefs, and empirical/social reliability |
| Persistence and reward memory | An agent retains or revises its current conditional action | Current assignments alongside accumulated propensities |
| Learning how to learn | An outer learner chooses an inner learning rule or adviser | An outer urn, fresh inner tasks, and an explicit return function |
| Generalization | Experience changes responses in related circumstances | A specified similarity relation and reward-spreading rule; retained as source work |

The **full-state representation** makes every table, option weight, assignment, reference point, clock, and current world state inspectable. Each game separately controls what its agents can observe. A receiver can therefore remain ignorant of nature even while a researcher inspects the entire simulation.

A game specifies six connected parts: environment, modules, decisions, interaction order, feedback, and experiment protocol. Feedback names its state targets explicitly. This accommodates repeated choices in dialogue, reciprocal rewards to an unsampled host option, and translator-only learning through frozen components. See [architecture](docs/architecture.md) for these contracts.

## Urns, randomness, and learning

An urn stores a nonnegative weight `q_i` for each available option. Ordinary drawing samples option `i` with probability

```text
p_i = q_i / sum_j(q_j)
```

Weights are numeric; fractional reinforcement requires no physical ball objects. Sampling leaves the stored weights intact. An update subsequently adds or removes the prescribed amount.

The draws make a trajectory stochastic. Nature, scheduling, noise, and randomized completion can supply further random choices. A checkpoint preserves both adaptive state and random-generator state, so continuation reproduces the trajectory within the pinned implementation. Persistent-choice learners also store a current assignment, which governs their response until it changes.

The **R1–R7 labels below are repository identifiers** for the rule families. They include numerical updates, structural invention, and nested learning. Each game determines which decisions receive feedback and when.

| Method | Mechanism | Additional state or conditions |
|---|---|---|
| **R1 — Simple reinforcement** | Add nonnegative payoff to the chosen option: `q_k' = q_k + payoff`; normalize weights to choose again | Standard signaling starts each option at one and pays one on success |
| **R2 — Reinforcement with punishment** | Add `a` on success; subtract `b` on failure, keeping the chosen weight at least one | Success is recorded separately from numerical payoff |
| **Signed payoff reinforcement** | Add net payoff to the chosen weight with floor one | Used for costs: an action can succeed while its cost produces a negative return |
| **R3 — Iterative punishment** | Alternate reward-only and punishment-only phases | A play clock and configured phase lengths determine the active update |
| **R4 — Adjustable-reference-point reinforcement** | Discount propensities, then update all options according to payoff relative to a reference point | A reference point, experience-spread parameter, positive floor, and separate upward/downward reference rates |
| **R5 — Reinforcement with invention** | A mutator proposes a signal; successful first use introduces it into its sender's contexts and reinforces its use | Dynamic signal identities, provisional receiver rows, and game-specific rollback rules |
| **R6 — Win-stay/lose-randomize with reinforcement** | Success retains and reinforces current conditional choices; failure resamples the visited assignments using accumulated weights | Persistent assignments remain distinct from propensity weights |
| **R7 — Outer reinforcement** | Reward a chosen learning rule or adviser using the outcome of its task | Outer weights persist while specified inner tasks reset |

For **R4**, let `R = payoff − old_reference`. The book's printed experience rule adds `(1−epsilon)R` to the chosen option and `epsilon R` to **each** unchosen option after discounting, subject to a small positive floor. The reference then moves toward the payoff. The runtime supports this literal update on single-vector bandit/adviser tasks; canonical use across multiple conditional urns needs a source decision about reference-point scope. The exact equations and parameters appear in [learning rules](docs/learning-rules.md), B11 §11.2.4.

In **R5**, a successful new token enters every input urn of its sender at weight one, and its origin context receives the ordinary reinforcement. Single-sender invention discards a failed proposal and its provisional receiver row. Pair-sender ordering and addition retain explicit choices about first-use failure handling. Forgetting can sample balls by frequency or sample represented types; these probability measures produce different dynamics.

Additional executable methods include **bounded reinforcement** with an explicit per-option ceiling, **discounting**, two-option **win-stay/lose-shift**, and **whole-strategy win-stay/lose-randomize**. The whole-strategy comparator replaces both full signaling maps on failure. R6 instead changes only the conditional assignments involved in that play. Exploration error, forgetting timing, and bounding policies are named profile choices.

Barrett also discusses probe-and-adjust, Bayesian conditioning, best response with inertia, and population replicator dynamics. The finite runtime now implements population replicator dynamics for supported game adapters. A related mean-field approximation describes the leading expected change under simple reinforcement; conditional visitation and growing urn totals affect its time scale. Bayesian conditioning and best response remain documented extensions. See B2 pp.25–27, the [method specifications](docs/learning-rules.md), and [finite dynamics](docs/finite-dynamics.md).

## The games

Each entry below names the task and the capacity it exercises. **Executable** means a runnable profile exists. **Variant** identifies operational choices that remain explicit pending source clarification. **Analysis** and **catalogued** entries preserve concepts and further models without presenting a complete runner. The [machine-readable catalogue](src/self_assembling_games/data/catalogue.json) tracks implementation, profile, and source-reproduction status separately.

### Signaling, representation, and control — B2–B4

| ID | Game and mechanism | Capacity and status |
|---|---|---|
| **SIG** | A sender observes nature, sends a signal, and a receiver selects an act; success means the act matches the state | Learned conventions; executable general `i×j×k`, equal-size, and biased-nature profiles |
| **COD** | Two senders each observe one of four states and send a binary term; a receiver acts on their ordered pair | Coordinated cross-partitions and joint coding; executable |
| **NAND** | Each sender observes one truth value; the receiver acts on their two signals to produce NAND | Coevolution of representation and a logical operation; executable |
| **HIER** | Basic senders see color and animal; an executive sender sees the requested context and activates A, B, or both; an executive receiver selects output interpretation | Efficient individual/composite expressions and hierarchical control; executable with signed signaling costs |
| **CUE** | The state-to-cue association is fixed while the receiver learns a useful response | Cue reading as a one-sided precursor; executable |
| **MAN** | Receiver dispositions are fixed while the sender learns which behavior triggers the desired act | Sensory manipulation as a one-sided precursor; executable |

In SIG, successful decisions can become a matched language, while some runs pool states under the same signal. The unbiased two-state R1 game has the special probability-one limiting result cited by Barrett. Larger or biased games can settle into suboptimal conventions. COD shows how two limited senders can jointly distinguish four states. HIER adds pressure to communicate only the information required by the context: one term for color or animal, two for both. See [signaling specifications](docs/games/signaling.md) and [precursors](docs/games/transfer.md).

### Rule following, transfer, and learned connections — B5–B6

| ID | Game and mechanism | Capacity and status |
|---|---|---|
| **ORD** | Two sender modules see ranked stimuli and invent terms; a receiver learns `<`, `>`, or `=` from their pair | Acquired ordering vocabulary; executable variant with explicit failed-invention handling |
| **TR-ORD** | New stimuli select old input urns of a frozen ordering game | Transfer of an acquired relation; executable with learned or constructed templates |
| **TR-PART** | Train ordering translators on adjacent pairs and evaluate nonadjacent comparisons | Inference under incomplete training; executable with an explicit pair sampler |
| **TR-LOG** | Binary translators select inputs of a frozen logical game, such as inverting both inputs of NAND to obtain OR | Logical template transfer; executable |
| **KIND** | Group stimuli by their learned conditional response distributions | Conventional concepts and kinds; available as policy analysis |
| **POLY** | Join successful interactions into a serial chain | Polymerization of games; catalogued proposal requiring timing and credit rules |
| **SCOD** | In COD, a receiver additionally learns whether to attend to A, B, or both | Self-assembling coding through learned salience; executable |

Template transfer preserves an acquired internal structure while translators learn how to use it for a new problem. Frozen modules continue to sample their learned policies. Salience learning chooses an information source; executive activation chooses which component acts. These decisions have different effects on observation and feedback. See [assembly and transfer specifications](docs/games/transfer.md).

### Computation and discourse — B7–B8

| ID | Game and mechanism | Capacity and status |
|---|---|---|
| **COMP-NAND** | Frozen NAND modules are joined by one or two learned source selectors to compute `(X NAND Y) NAND Q` | Computational composition through routing; executable |
| **ADD** | Two sender modules observe cardinalities `0…5` and invent terms; the receiver chooses a sum `0…10` | Acquired arithmetic representation; executable invention variant |
| **ADD-COMP** | A frozen addition output is translated into an input of a frozen comparison game | Arithmetic followed by comparison; executable with explicit tie and template choices |
| **ASK** | An uninformed assistant can guess slab/block or ask an informed builder, then act or ask again | Initiating discourse and deciding when to stop; executable exchange-cost variant |
| **DIAL** | An assistant asks up to two questions about one of four materials; the informed builder supplies one of four answers | Learned questions, answers, and turn structure; executable reward/punishment profiles |
| **DIAL-restricted** | DIAL permits only two answer types | Complementary questions needed for four-way distinction; executable |
| **DIAL-costly** | Continued dialogue incurs an exchange cost against a success reward | Learning economical discourse; executable history/cost variant with a censoring cap |

Composition preserves representation adapters and credit recipients: pretrained computation remains fixed while a selector or translator learns. A shared upstream invocation produces one stochastic output reused by its consumers. Dialogue preserves every decision occurrence, including repeated use of the same urn. ASK and costly dialogue record reaching their engineering turn cap as censoring. See [computation and discourse specifications](docs/games/computation-discourse.md).

### Networks and inquiry — B9–B10

| ID | Game and mechanism | Capacity and status |
|---|---|---|
| **VIS** | Each agent selects a host and reinforces its visiting choice using the interaction payoff; reciprocal profiles also update the host's option for the visitor | Formation of weighted directed networks; executable |
| **STAG** | Fixed stag/hare types select partners under their game payoffs | Partner selection conditional on strategic compatibility; executable fixed-type case |
| **FFSR** | Visitors and hosts draw complete signaling plans; expected success across states and both sender roles rewards plans and the visitor's link | Coevolution of conventions and partners; executable forgetting/error variant |
| **SUB** | Agents pay for outgoing information links and benefit from reachable information | Subscription networks; catalogued, with source-rule work for reinforcement and a best-response comparator |
| **DISC** | Bargaining responses and partner changes can produce persistent group inequity | Evolving bargaining networks; catalogued separate best-response dynamics |
| **INQ** | Inquirers choose nature, themselves, or another inquirer; true resulting beliefs reinforce the chosen source | Epistemic networks, empirical specialization, and information control; executable |

INQ updates beliefs immediately in a randomly ordered consultation round. Empirical reliability governs nature consultations, and the consulting agent's social reliability governs communication. Self-consultation retains its belief. Beliefs reset between inquiry problems while consultation weights survive. These timing and persistence choices help determine which agents become sources for others. See [network and inquiry specifications](docs/games/networks-inquiry.md).

### Adaptation, invention, and learning to learn — B11–B12

| ID | Game and mechanism | Capacity and status |
|---|---|---|
| **BOX** | Choose between two boxes with a fixed rewarded location or randomly replenished food | Comparing reinforcement with win-stay/lose-shift; executable |
| **CHANGE** | Carry a learned coding game through a changed state-to-action payoff map | Adaptation through bounded commitment; executable bounding variant |
| **INV** | Sender urns begin with a mutator and grow a vocabulary through successful first use | Signal invention; executable base and ball/type-forgetting variants |
| **HYB** | SIG uses persistent conditional assignments and reinforcement-weighted resampling after failure | Combining persistence with reward memory through R6; executable |
| **BAND** | Choose a machine, receive its random payoff, and reinforce the choice | Learning from differential returns; executable with supplied reward distributions and R1/R4 profiles |
| **META-RULE** | An outer learner selects a rule for a fresh inner task and receives a task-based return | Learning which learning method suits a problem; executable specified-task profiles |
| **META-ORACLE** | An outer learner selects an adviser and reinforces correct answers, with a specified return for incorrect answers | Learning information providers; executable answer-process variant |
| **MH** | Choose stay or switch after a host reveals an eligible goat door | Monty Hall as a reinforced decision; executable with an explicit host protocol |
| **GEN** | Spread reinforcement across similar circumstances or options | Generalization; catalogued pending an operational similarity rule |
| **SET** | Improve learning behavior across repeated problems or tune update strengths | Learning sets and self-tuning; catalogued empirical/proposed mechanisms |
| **ATT-R** | A receiver chooses among the sender's signaling feature and spurious features before choosing an act | Learned receiver attention; executable with explicit distractor sampling |
| **ATT-B** | Sender and receiver both select observations before signaling and acting | Coevolving attention; executable source/value-specific and shared-value table variants |
| **RIT** | Saliences, actions, and dispositions to update them can become organized together | Ritualization of learning itself; catalogued explanatory proposal requiring primitive dynamics |

These tasks expose how a method's usefulness depends on the problem. The BOX comparisons motivate choosing a learner across fresh tasks. CHANGE tests whether accumulated experience leaves room for adaptation. Attention games let feedback select informative features while signal meanings coevolve. See [adaptive learning specifications](docs/games/adaptive-learning.md).

## How components compose

The library supports several distinct assembly operations:

1. **Parallel expression:** combine sender-specific terms into an ordered receiver input, as in COD.
2. **Serial computation:** pass an acquired output into another component through a declared representation adapter, as in ADD-COMP.
3. **Source selection:** learn whether to use a nature feature or an upstream output, as in SCOD and COMP-NAND.
4. **Template transfer:** translate new stimuli into the input space of frozen acquired modules, as in TR-ORD and TR-LOG.
5. **Executive control:** select active components and output interpretations from context, as in HIER.
6. **Sequential interaction:** learn to initiate, continue, and terminate a conversation, as in ASK and DIAL.
7. **Social connection:** select interaction partners or information sources, as in VIS and INQ.
8. **Structural invention:** introduce options and dependent decision contexts, as in INV, ORD, and ADD.
9. **Nested learning:** select an inner learning process while preserving the outer learner across tasks, as in META-RULE.

Each operation preserves observation restrictions, update targets, execution order, and reset boundaries. A learned template import copies its policy into distinct frozen instances and records training lineage. Invented vocabularies need explicit export choices for mutator suppression and missing pair rows. See the [runtime guide](docs/runtime.md) for artifact transfer and the [architecture](docs/architecture.md) for the structural contracts.

## Run and inspect a game

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/sg list
.venv/bin/sg list --all
.venv/bin/sg describe HIER
.venv/bin/sg run sig.n2.demo --seed 0 --plays 10000 --output runs/signaling
.venv/bin/sg inspect runs/signaling
.venv/bin/sg evaluate runs/signaling
.venv/bin/sg resume runs/signaling --plays 10000
```

`list` shows profiles; `list --all` shows the complete catalogue, including analytical and proposed entries. `describe` exposes a model or profile. Profiles specify domains, distributions, update rules, costs, schedules, and source choices. `run` creates a fresh output, while `resume` advances an existing checkpoint.

```python
from self_assembling_games import build, evaluate

simulation = build("hier.demo", seed=0)
simulation.run(plays=10_000)
state = simulation.inspect()
assessment = evaluate(simulation.snapshot())
simulation.checkpoint("runs/hierarchy")
```

Examples include `cod.demo`, `nand.demo`, `scod.demo`, `comp-nand.2.demo`, `ord.demo`, `dial-restricted.demo`, `inq.demo`, `band.arp.demo`, and `meta-rule.demo`. The [runtime guide](docs/runtime.md) gives the full workflow for learned templates, held-out tests, and custom profiles.

## Evaluation and experiments

The package separates **observed learning performance** from **current-policy evaluation**. Cumulative accuracy measures completed outcomes across training; a bounded window measures recent outcomes. Exact evaluation enumerates finite interaction paths under fixed policies. Sampled evaluation runs a copied environment with adaptive updates disabled and separate randomness. Ordering transfer also supports held-out nonadjacent comparisons.

Reports include accuracy, net payoff, terms, exchanges, outcome status, and completion/truncation where applicable. Censored or resource-limited interactions have explicit counts. Complete checkpoints preserve tables and their visitation order, learner and world state, clocks, novelty identifiers, metrics, and random-generator state. Optional traces record chance, decision, feedback, and structural events for replay.

```sh
.venv/bin/sg ensemble sig.n2.demo --seeds 0 1 2 3 --plays 10000 \
  --threshold 0.95 --workers 2 --output runs/ensemble
.venv/bin/sg benchmark hier.demo --plays 10000
.venv/bin/sg list --experiments
.venv/bin/python -m pytest -q
```

Ensembles fix seed sets and protocols, resume interrupted runs, and report distributions with bootstrap and Wilson confidence intervals. Independent workers preserve each seed's serial trajectory. Book-horizon profiles include `sig.n4.book` and `sig.n8.punishment.book`. Their registered long statistical reproductions await execution; source-reported percentages are documented reference results.

The [Barrett qualification report](docs/qualification.md) records its original **108 passing tests**, formatting/lint checks, wheel and source builds, installed CLI checks, and serial timing measurements. The finite dynamics addition extends verification to source strategy spaces, exact game fixtures, role-grouped fitness, selection and mutation equations, numerical integration, reinforcement continuation, and exact policy comparisons. Current validation is recorded in [finite dynamics](docs/finite-dynamics.md).

## Source fidelity and further work

A demonstration supplies an executable configuration and labels its choices. A source reproduction additionally requires source parameters, an explicit measurement protocol, and resolved operational questions. The runtime gates canonical profiles affected by unresolved source details; [review](docs/review.md) tracks Q1–Q13 and the relevant follow-up sources. The catalogue retains qualitative mechanisms and separate dynamics so their place in the book remains visible.

The common abstraction and composition operations are engineering designs derived from the games. “Category” here means the organized family of models and operations. A formal mathematical category would require separately specified objects, morphisms, and laws.

For further reading, use the [catalogue](docs/catalogue.md), [learning rules](docs/learning-rules.md), [architecture](docs/architecture.md), [source register](docs/sources.md), [Astra review](docs/astra-review.md), and [repository plan](docs/repository-plan.md). The five game specification sheets expand the mechanisms and chapter citations presented here.
