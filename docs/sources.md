# Sources and quotation anchors

## Primary book

Barrett, Jeffrey A. (2025). *Self-Assembling Games: Language, Learning, and Inquiry*. Oxford University Press. The chapter PDFs identify the short book title as *Self-Assembling Games*. Citations below use the book's printed page numbers, which are continuous across chapters.

The register covers all thirteen chapters, using printed page intervals and chapter DOI links. The repository distributes bibliographic references and model specifications.

| ID | Chapter title | Printed pages | DOI |
|---|---|---|---|
| B1 | Self-Assembly and Games | 1–12 | [Chapter 1](https://doi.org/10.1093/9780197793572.003.0001) |
| B2 | Learning to Signal | 13–37 | [Chapter 2](https://doi.org/10.1093/9780197793572.003.0002) |
| B3 | Generalized Signaling Games | 38–55 | [Chapter 3](https://doi.org/10.1093/9780197793572.003.0003) |
| B4 | Assembling Games from Decisions | 56–69 | [Chapter 4](https://doi.org/10.1093/9780197793572.003.0004) |
| B5 | Rule Following and Template Transfer | 70–96 | [Chapter 5](https://doi.org/10.1093/9780197793572.003.0005) |
| B6 | Modular Composition and the Evolution of Salience | 97–111 | [Chapter 6](https://doi.org/10.1093/9780197793572.003.0006) |
| B7 | The Self-Assembly of Computation | 112–128 | [Chapter 7](https://doi.org/10.1093/9780197793572.003.0007) |
| B8 | Learning to Take Turns | 129–147 | [Chapter 8](https://doi.org/10.1093/9780197793572.003.0008) |
| B9 | The Evolution of Networks | 148–168 | [Chapter 9](https://doi.org/10.1093/9780197793572.003.0009) |
| B10 | The Self-Assembly of Inquiry | 169–185 | [Chapter 10](https://doi.org/10.1093/9780197793572.003.0010) |
| B11 | The Ways of Learning | 186–232 | [Chapter 11](https://doi.org/10.1093/9780197793572.003.0011) |
| B12 | Learning How to Learn | 233–270 | [Chapter 12](https://doi.org/10.1093/9780197793572.003.0012) |
| B13 | Self-Assembling Games and Explanation | 271–276 | [Chapter 13](https://doi.org/10.1093/9780197793572.003.0013) |

## Model definitions and quotation anchors

These short anchors accompany the original paraphrased specifications in this repository.

**Conditional acts as the basic unit.** Barrett states:

> “We will adopt an act-based version of reinforcement learning and use balls and urns to represent agent propensities.”

[B2 §2.2, p.25](https://doi.org/10.1093/9780197793572.003.0002). The core therefore stores a propensity table for each observation context. Complete strategy urns are used separately where a later model explicitly requires them.

**Which decisions learn in basic signaling.**

> “The sender updates her dispositions to signal conditional on the state and the receiver updates his dispositions to act conditional on the signal.”

[B2 §2.3, p.29](https://doi.org/10.1093/9780197793572.003.0002). Feedback goes to the currently visited sender and receiver rows.

**Single-term interpretation in hierarchical composition.**

> “If the expression contains just one term, he randomizes with uniform probabilities between the two urns corresponding to the sender and the term sent and draws a ball from that urn.”

[B3 §3.3, p.49](https://doi.org/10.1093/9780197793572.003.0003). The receiver retains four pair urns and completes a missing component by chance.

**Preserving options under cost-driven punishment.**

> “If the number of balls of a type in any urn goes below one, we will set it to one to keep that type represented in the urn.”

[B3 §3.3, p.50](https://doi.org/10.1093/9780197793572.003.0003). This supports the chosen-option floor in HIER's signed updates.

**The mechanism binding modules.** Barrett describes modular composition through the

> “evolution of new saliences through ritualization”

[B6 §6.2, p.103](https://doi.org/10.1093/9780197793572.003.0006). Source-selection and translation decisions are concrete ways to implement the learned connection.

**What remains adaptive during the NAND composition example.**

> “P’s saliences are the only dispositions of the composite system that evolve during play.”

[B7 §7.1, p.114](https://doi.org/10.1093/9780197793572.003.0007). The internal NAND policies remain fixed while P learns which input to use.

## Cited antecedents and clarification targets

The references in this table are **Barrett's attributions**, as located in the named chapter. The model definitions follow the book; open operational questions identify the original sources needed to resolve them.

| Names/year as cited | Book location | Role |
|---|---|---|
| Lewis (1969); Skyrms (2010) | B2 §2.3 | Classical conventions and low-rationality signaling |
| Roth and Erev (1995); Erev and Roth (1998) | B2 §2.2 | Propensity reinforcement and response rule |
| Argiento et al. (2009) | B2 p.30 n.22 | Limiting two-state signaling result |
| Barrett (2006); Huttegger et al. (2010) | B2 pp.31–34 | Pooling, biased states, larger games |
| Barrett and Gabriel (2022) | B2 p.34 n.28 | Larger signaling games and learning dynamics |
| Barrett (2007) | B3 §3.1; B3 §3.3 | Coding and compositional language |
| Barrett, Skyrms and Cochran (2018); Barrett et al. (2020) | B3 §3.3 | Hierarchical control and composition |
| Barrett and Skyrms (2017) | B2 §2.1; B4 discussion | Self-assembly from decisions |
| Barrett (2023a) | B6; B7; B12 §12.6 | Salience and modular assembly |
| Skyrms and Pemantle (2000) | B9 §9.2 | Reinforced network formation |
| Fulker, Forber, Smead and Riedl (2024) | B9 §9.4 | Signaling strategies and visiting networks |
| Bala and Goyal (2000) | B9 §9.5 | Information subscription networks |
| Rubin and O'Connor (2018) | B9 §9.6 | Discrimination in bargaining networks |
| Bereby-Meyer and Erev (1998) | B11 §11.2.4 | ARP experience/reference-point updates |
| Barrett (2014) | B11 §11.2.5 | Bounded learning and changing codes |
| Alexander, Skyrms and Zabell (2012) | B11 §11.2.6 | Reinforcement with invention and forgetting |
| Barrett et al. (2017) | B11 §11.3.2 | Hybrid WSLRwR |
| O'Connor (2017); Smead (2015) | B12 §§12.4–12.5 | Generalization and evolution of learning |
| Torsell and Barrett (2024) | B12 §12.5 | Self-tuning learning discussion |
| Herrmann and VanDrunen (2025) | B12 §12.6 | Attention games |
| Beggs (2005); Hopkins and Posch (2005); Börgers and Sarin (1997) | B2 pp.25–27; B12 §12.2 | Reinforcement, mean-field, replicator connections |

## Source questions affecting profiles

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

Source-supported resolutions belong with the corresponding model and profile. Demonstrations state their operational choices explicitly.

## Transcription notes

The finite Skyrms family is registered from [Skyrms (2009), §§2–5](https://doi.org/10.1098/rstb.2008.0258). Its game definitions, equations, table discrepancies, dialogue information-set question, and supplementary-protocol gate are recorded in [finite dynamics](finite-dynamics.md). Source fidelity is recorded separately for the game, adaptive method, population specification, and numerical or encounter protocol.

- Printed page numbers take priority over incomplete catalogue page fields. Repeated author/book-author metadata is normalized to one bibliographic author in this document.
- B3's coding illustration mentions state 4 within a four-state `0…3` domain; the game sheet identifies the correction.
- B11's ARP equations contain inconsistent payoff indices and an APR spelling. Its unchosen-option update is transcribed literally in the rule sheet pending original-source clarification.
- B3's repeated .975 phrase in the hierarchical footnote is treated as one reported proportion, not two experiments.
- Figures were consulted alongside extracted text where conditional structure matters, including the hierarchical composition figure. Future corrections should name the precise source page or figure and the resulting model change.

## Multilocus population methods and constructed assays

The multilocus extension follows standard haploid selection/transmission
recursions and multinomial Wright–Fisher sampling. Its [method specification](multilocus.md)
records the precise equations, census convention, expression map, participation
bindings and life-cycle order.

- **Kirkpatrick, M., Johnson, T., & Barton, N. H. (2002).** General models of
  multilocus evolution. *Genetics*, 161(4), 1727–1750.
  [doi:10.1093/genetics/161.4.1727](https://doi.org/10.1093/genetics/161.4.1727).
  The institutional/publisher abstract was checked for the scope of the general
  multilocus framework. The elementary operators implemented here are specified
  independently in the method guide.
- **Barton, N. H., & Otto, S. P. (2005).** Evolution of Recombination Due to Random
  Drift. *Genetics*, 169(4), 2353–2370.
  [doi:10.1534/genetics.104.032821](https://doi.org/10.1534/genetics.104.032821);
  [author PDF](https://www.zoology.ubc.ca/~otto/Reprints/BartonOtto2005.pdf).
  The model description and Appendix A were examined for haploid selection,
  random mating/meiosis and multinomial juvenile sampling. Their `2N` haploid
  sample corresponds to this library's `population_size=N` haploid genomes.
- **Supplied working study (11 October 2026):** *Material realisations of
  self-assembling games: Organisation, inheritance, and the constitution of
  adaptive capacities*. The user-supplied text defines the toy coordination,
  fixed-NAND adapter and parity-composition examples and the single-boundary
  exchange operator. The library's three `ML-*` kernels and expression tables
  implement those constructed assays. Their source status is
  `explicit_constructed_assay`; the supplied template and preformed module
  assumptions are retained in game fidelity records. No published biological
  findings are attributed to this working study.

Independent interval switches, per-locus mutation matrices, explicit affine
reproductive weights and encounter sampling with replacement are declared model
choices. Barrett/Skyrms source-model observations and adaptive laws retain their
existing definitions and source gates. The multilocus toy demonstrations are
listed separately through `sg list --family constructed`.

## Situated colour-board coordination

Scott-Phillips, T. C., Kirby, S., & Ritchie, G. R. S. (2009). *Signalling
signalhood and the emergence of communication*. Cognition.
[doi:10.1016/j.cognition.2009.08.009](https://doi.org/10.1016/j.cognition.2009.08.009).
The [situated interaction guide](situated-interactions.md) specifies `ecg.demo` as
a constructed discrete-time variant with private colour boards, ordinary movement
and finish actions, synchronous ticks and an explicit truncation cap. Random and
scripted local controllers qualify its mechanics; human signalhood recognition
and learned communication remain research questions for further models.
