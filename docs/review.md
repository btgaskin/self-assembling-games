# Source questions and mechanism checks

## Adopted decisions

- Keep this as a standalone local research repository.
- Use Barrett's book models as the canonical family, with source-specific observations, timing, updates, and parameters.
- Build a small Python/NumPy implementation after this structure is reviewed.
- Use explicit conditional urns, ordered decision records, and feedback assignment as the common core.
- Keep activation, salience, translation, invention, dialogue, partner selection, inquiry, and meta-learning as explicit composition mechanisms.
- Inspect the full state of every module while restricting each agent to its source-defined observations.
- Retain source profiles and deliberately small demonstration profiles separately.
- Start with reinforcement. Keep the game definitions suitable for later mean-field/replicator work.

## Source questions affecting canonical profiles

| Question | Source location | What needs resolution | Consequence |
|---|---|---|---|
| Q1 — Long/costly discourse | B8 §§8.2–8.3, pp.133–145 | Exchange versus individual-message cost; repeated-Q reward; negative returns; unlimited-history storage | Avoid importing HIER's signed floor silently into a simple-reinforcement dialogue |
| Q2 — ARP across conditional decisions | B11 §11.2.4, pp.204–206; Bereby-Meyer and Erev (1998) | Per-unchosen experience increment; reference-point scope; forgetting scope; initial q | Exact printed two-option rule is clear; multi-option and multi-urn adaptation needs a chosen source profile |
| Q3 — Invention in pair games | B5 §5.4; B7 §7.3; Alexander et al. (2012) | Receiver pair-row fate after failed new terms; joint novelty transaction; forgetting mutators/extinction | Dynamic state and subsequent exploration depend on retention/cleanup |
| Q4 — Addition/comparison staging | B7 pp.121–125 | Old comparison pretraining; ties; frozen checkpoint criterion | Reproduction needs the exact staged protocol |
| Q5 — Logical-transfer closure | B5 §5.8 and its template discussion | Which input/output permutations/translations are allowed | Functions with the same false-row count need not be equivalent under independent input recodings |
| Q6 — FFSR exploration | B9 §9.4; Fulker et al. (2024) | Forgetting/error rates, order, strategy draws and visit schedule | Determines network/convention persistence |
| Q7 — Bounded/self-tuning learners | B11 §§11.2.3,11.2.5; Barrett (2014); B12 §12.5 | Bound operation, numeric parameters, phase-adjustment criterion | Avoid conflating count ceilings, total bounds, and discounting |
| Q8 — Attention extension | B12 §12.6; Herrmann and VanDrunen (2025) | Distractor sampling/correlation and sender conditional-table sharing | Defines what information is available and which policies can evolve |
| Q9 — Nested task environments | B12 §§12.2–12.3 | Bandit distributions; trial parameters; positive failure return; oracle processes | Enables reproducible meta-learning comparisons |
| Q10 — Qualitative assembly/network extensions | B4, B6 §6.1, B9 §§9.3,9.5–9.6, B12 §§12.4–12.7 | Primitive dynamics, imitation rate, subscription learner, generalization kernel | These are framework entries until a concrete source protocol is recovered |
| Q11 — Partial ordering sampler | B5 §5.6 | Exact adjacent training-pair selection and held-out protocol | Demonstrations sample adjacent pairs in both directions explicitly |
| Q12 — Frozen template lifecycle | B5, B7 | Pretraining budget/checkpoint selection; frozen mutator behavior; unseen receiver pairs | Export transforms and their lineage are explicit variants |
| Q13 — Monty Hall host | B11 §11.4 | Exact host protocol | Demonstration opens a uniformly selected eligible goat door |

These questions concern specific profiles, so they need not delay the fully specified finite signaling core. Recover the cited papers when a corresponding profile enters implementation. Record source-supported resolutions in the register; preserve a literal book profile if a correction is needed.

## Checks that future implementation should satisfy

| Mechanism | Concrete check |
|---|---|
| Proportional response | Every urn normalizes; equal weights give equal probabilities |
| Feedback selection | Update footprints equal the declared credit rule, including unsampled recipients; frozen adaptive state remains byte-for-byte stable |
| Punishment | Floor holds after negative reward; other types retain their previous weights |
| Perfect finite policy | Exact success equals one under a matching convention |
| NAND baseline | Always-T policy evaluates to .75 under independent fair inputs |
| Coding | Correct cross-partition and receiver table identify all four states |
| HIER | One-term completion uses either compatible receiver row with probability .5; optimal mean terms `4/3` and net reward one |
| Invention | Success commits all required options/rows; failure follows source rollback; IDs remain distinguishable |
| Dialogue | Repeated draws preserve occurrence order; correct history keys and source termination rule are used |
| FFSR | Complete plans number sixteen; expected payoff averages states and both roles |
| Inquiry | Self is eligible; self preserves current truth; social error uses consulter reliability; weights persist across problem resets |
| Meta-learning | Each fresh inner problem resets its state while outer weights persist |
| Evaluation | Held-out/frozen evaluation produces no updates; cumulative and current-policy metrics remain separate |
| Reproducibility | Checkpoint restores learner state, clocks, provisional IDs, and random generator state |

These checks guide the executable test suite; actual results are recorded in the qualification report. Statistical convergence comparisons should follow deterministic rule checks. Include seed counts, run lengths, cutoff definitions, distribution of outcomes, and baseline policies with any future report.

## Review boundary

The documentation is ready for structural review: game coverage, source fidelity, common contracts, composition operations, and implementation order. The approved plan has been implemented as a reinforcement runtime with explicit source gates. See [runtime.md](runtime.md), [Astra review](astra-review.md), and [repository plan](repository-plan.md). Population/replicator dynamics and remote publication remain subsequent work.
