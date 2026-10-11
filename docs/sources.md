# Sources and quotation anchors

## Primary book

Barrett, Jeffrey A. (2025). *Self-Assembling Games: Language, Learning, and Inquiry*. Oxford University Press. The chapter PDFs identify the short book title as *Self-Assembling Games*. Citations below use the book's printed page numbers, which are continuous across chapters.

All thirteen chapters were read from the user's Zotero attachments for this specification. Chapters 2 and 3 were also supplied in the conversation. The register records selected parent items and attachment fingerprints. Duplicate chapter records were left in place. No PDFs or full extracted chapter text are distributed in this repository.

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

## Quotations tied to implementation decisions

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

The references in this table are **Barrett's attributions**, as located in the named chapter. The present milestone derives rules from the book; it does not claim an independent reproduction or complete reading of each antecedent paper. Retrieve the relevant original before resolving a parameter that the book leaves open. Full bibliographic metadata and paper identifiers can be added when those sources are read.

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

## Provenance and transcription notes

The finite Skyrms family is registered from [Skyrms (2009), §§2–5](https://doi.org/10.1098/rstb.2008.0258). Its game definitions, equations, table discrepancies, dialogue information-set question, and supplementary-protocol gate are recorded in [finite dynamics](finite-dynamics.md). Source fidelity is recorded separately for the game, adaptive method, population specification, and numerical or encounter protocol.

- [chapters.json](../sources/chapters.json) records the chosen Zotero items, attachments, PDF SHA-256 fingerprints, and verified printed page intervals. Local filesystem paths are omitted.
- Duplicate parents were found for B5 (`BZUAI2BS`, attachment `LA27DJDS`) and B6 (`NGPK74Z9`, attachment `ZP9XS85L`). Selected parents are recorded in the manifest; library records were preserved.
- Printed page numbers take priority over incomplete catalogue page fields. Repeated author/book-author metadata is normalized to one bibliographic author in this document.
- B3's coding illustration mentions state 4 within a four-state `0…3` domain; the game sheet identifies the correction.
- B11's ARP equations contain inconsistent payoff indices and an APR spelling. Its unchosen-option update is transcribed literally in the rule sheet pending original-source clarification.
- B3's repeated .975 phrase in the hierarchical footnote is treated as one reported proportion, not two experiments.
- Figures were consulted alongside extracted text where conditional structure matters, including the hierarchical composition figure. Future corrections should name the precise source page or figure and the resulting model change.
