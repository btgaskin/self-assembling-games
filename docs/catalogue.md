# Catalogue of games and assembly mechanisms

This catalogue covers the supplied book's thirteen chapters. “Specified” means the book supports a concrete mechanism at the level described in the linked sheet. It does not mean a runtime exists. “Clarify” identifies an operational detail or numerical profile to recover from a cited source. “Proposal” identifies a qualitative mechanism. “Comparator” and “later dynamics” retain related material in the structure without folding it into ordinary urn reinforcement.

## Chapter map

| Chapter | Role in the framework | Specification |
|---|---|---|
| B1 — Self-Assembly and Games | Decisions, adaptive dispositions, explanatory aim | [Architecture](architecture.md) |
| B2 — Learning to Signal | Core learner; simple signaling; pooling; alternative dynamics | [Signaling](games/signaling.md), [rules](learning-rules.md) |
| B3 — Generalized Signaling Games | Coding, NAND, hierarchical expression and control | [Signaling](games/signaling.md) |
| B4 — Assembling Games from Decisions | Ritualization, cue reading, sensory manipulation | [Transfer](games/transfer.md) |
| B5 — Rule Following and Template Transfer | Ordering, analogy, incomplete training, logical transfer | [Transfer](games/transfer.md) |
| B6 — Modular Composition and the Evolution of Salience | Chains, source selection, self-assembling coding | [Transfer](games/transfer.md), [architecture](architecture.md) |
| B7 — The Self-Assembly of Computation | Composed NAND; arithmetic and comparison | [Computation](games/computation-discourse.md) |
| B8 — Learning to Take Turns | Questions, dialogue, learned timing | [Discourse](games/computation-discourse.md) |
| B9 — The Evolution of Networks | Partner choice, strategies, subscriptions and bargaining | [Networks](games/networks-inquiry.md) |
| B10 — The Self-Assembly of Inquiry | Empirical and social source choice | [Inquiry](games/networks-inquiry.md) |
| B11 — The Ways of Learning | Punishment, phases, reference points, invention, hybrids | [Rules](learning-rules.md), [adaptive learning](games/adaptive-learning.md) |
| B12 — Learning How to Learn | Bandits, meta-learning, generalization, attention | [Adaptive learning](games/adaptive-learning.md) |
| B13 — Self-Assembling Games and Explanation | Interpretive synthesis and how-possible explanations | [Architecture](architecture.md), [review](review.md) |

## Stable model identifiers

| ID | Model or mechanism | Adaptive choices | Learning | Readiness |
|---|---|---|---|---|
| SIG | Lewis–Skyrms `n×n×n`; general `i×j×k`; biased nature | Signals and acts | R1, configurable R2/R3/R6 | Specified |
| COD | Four-state coding | Two signals, joint act | R1/R2 | Specified |
| NAND | Binary truth-function game | Two signals, truth act | R1/R2 | Specified |
| HIER | Hierarchical composition | Activation, signals, interpretation, act | Signed reward, floor one | Specified |
| CUE | Cue reading | Receiver response | R1; fixed sender | Specified precursor |
| MAN | Sensory manipulation | Sender behavior | R1; fixed receiver | Specified precursor |
| ORD | Seven-stimulus ordering | Invented signals and relation | R5 | Clarify first-failure rows |
| TR-ORD | New ordering via old template | Old input selection | R2; frozen template | Specified |
| TR-PART | Adjacent-only transfer training | Old input selection | R2 then fixed evaluation | Clarify exact pair sampler |
| TR-LOG | NAND/OR template transfer | Old truth-input selection | R1; frozen template | Specified; broader closure claim open |
| KIND | Conventional concepts/kinds | Analysis of learned partitions | Existing game learner | Interpretation/analysis |
| POLY | Polymerization of interactions | Candidate connections | Task-dependent | Proposal |
| SCOD | Self-assembling coding | Signals, attention, act | R1 | Specified |
| COMP-NAND | One- and two-selector computational composition | Module input source | R1; frozen modules | Specified |
| ADD | Small-cardinal addition | Invented signals, sum act | R5 | Clarify source profile |
| ADD-COMP | Addition–comparison composition | Translation into old comparison | R2; frozen modules | Clarify ties/pretraining |
| ASK | Slab/block question | Start/continue/act, answer | R1, exchange reward | Clarify repeated-question costs |
| DIAL | Four-material dialogue | Questions, replies, action | R1/R2 | Specified finite variant |
| DIAL-restricted | Two-answer dialogue | Same, two answers | R1 | Specified finite variant |
| DIAL-costly | Costly unlimited dialogue | Continue/stop | Cost-adjusted feedback | Clarify history and negative returns |
| VIS | Visiting networks | Host | Payoff reinforcement | Specified core; game supplies payoff |
| STAG | Fixed stag/hare visiting | Host | Reciprocal reinforcement | Specified; imitation extension clarify |
| FFSR | Signaling/partner coevolution | Host and complete strategy | Expected payoff + exploration | Clarify forgetting/error parameters |
| SUB | Information subscriptions | Link set | Source-dependent | Clarify RL; best-response comparator |
| DISC | Bargaining and discrimination | Partners and bargaining strategy | Best response | Later dynamics |
| INQ | Epistemic-network inquiry | Nature, self, other agent | R1 on true belief | Specified |
| BOX | Fixed/random reward boxes | Box | R1; WSLS comparator | Specified |
| CHANGE | Sequentially changing coding payoff | Signal partitions/interpretation | Bounded reinforcement | Clarify bounding rule |
| INV | Signal invention/forgetting | Signal, new type, act | R5 plus forgetting | Base specified; forgetting clarify |
| HYB | WSLRwR signaling | Current assignments on failure | R6 | Specified |
| BAND | Reinforced bandits | Machine | R1 weighted by payoff | Distribution supplied per experiment |
| META-RULE | Learning-rule selection | Inner learner | R7 | Specify trial profile/positive failure return |
| META-ORACLE | Adviser selection | Oracle | R7 | Specify answer process |
| MH | Monty Hall | Stay/switch | R1 or Bayesian comparator | Specify host protocol |
| GEN | Generalization | Similar circumstances/options | Spread reinforcement | Clarify similarity rule |
| SET | Learning sets and self-tuning | Rule strength/salience | Adaptive learning schedule | Proposal/cited clarification |
| ATT-R | Receiver attention | Signal, feature, act | R1 | Specified; pin distractor law |
| ATT-B | Both agents learn attention | Source/feature, signal, act | R1 | Clarify table sharing/distractor law |
| RIT | Ritualization of learning | Saliences and adaptive dispositions | Primitive dynamics unspecified | Proposal |

These IDs are repository labels. Source names and attributions remain in the game sheets. Families share mechanisms; they need not each become a separate class. For example, SIG with R3 and META-RULE choosing R3 are different experiments using the same learner.

## A useful ordering for executable work

1. Finite full-state urns and R1/R2: SIG, COD, NAND, HIER.
2. Structural breadth and frozen composition: SCOD, COMP-NAND, TR-LOG, dialogue, networks, inquiry.
3. Provisional invention and dynamic rows: INV, ORD, ADD, then learned TR-ORD/TR-PART/ADD-COMP.
4. Turn-based histories: finite DIAL, then source-resolved ASK/costly dialogue.
5. Partner and consultation schedulers: VIS/STAG/FFSR, INQ.
6. Extra learner state and nested tasks: R3/R4/R6, BOX/BAND/META, ATT-R/ATT-B.

This is a proposed dependency order for review. The runtime and its current executable/source statuses are described in [runtime.md](runtime.md). The machine-readable registry separately tracks mechanism, profile, implementation, and source-reproduction status.
