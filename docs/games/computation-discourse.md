# Computation and discourse

## COMP-NAND — Composition of pretrained computational modules

**Source:** [B7 §7.1, pp.112–116](https://doi.org/10.1093/9780197793572.003.0007).

Pretrain two NAND games and freeze their internal urns. Nature supplies independent unbiased truth values X, Y, Q, and an irrelevant candidate Z. The desired output is `(X NAND Y) NAND Q`. Module 1 receives X,Y and produces its learned truth output. One input of module 2 is Q; its other input P has a salience urn choosing Z or module 1's output, each initially weight one.

Evaluate the full network output against the desired truth function. Reward only P's selected source by one on success; failure pays zero. The pretrained modules draw from frozen policies. A source option is a wiring decision, not a replacement NAND truth table. Ensure the output act's truth interpretation matches the receiving module's input interpretation.

In the two-selector variant, P chooses Z or module 1's output and the other input selector chooses Q or module 1's output. Reward both source selections on successful full computation; keep all internal modules frozen. The useful route is then jointly selected. The direct scratch comparator uses three truth-input senders and a receiver acting on their signal triple; its observations match X,Y,Q and it learns the composite function from scratch.

## ADD — Inventing an addition code

**Source:** [B7 §7.3, pp.120–124, Fig.7.2](https://doi.org/10.1093/9780197793572.003.0007).

Sender A sees one cardinality from zero through five; sender B sees another, each selected randomly with equal probabilities. Each has six urns initially containing one mutator, which remains represented by precisely one ball in each urn. Signals are invented as in ordering. The receiver has a dynamically created urn for each ordered signal pair, with eleven sum actions `0…10`, initially one each. A successful play is an act equal to `a+b`.

Existing successful draws gain one. A successfully invented term is added to all input urns of its sender and reinforced in its current context. Failure supplies no positive reinforcement. Failed-invention receiver-row retention should be pinned from the cited implementation before a source-reproduction profile is run. With eight million plays, the book reports fractions .67/.83/.95 of runs exceeding cumulative accuracy .95/.90/.85 respectively. The domain, uniform cardinality sampling, sum table, and novelty mechanism define the learning problem.

## ADD-COMP — Addition followed by comparison

**Source:** [B7 §7.3, pp.123–125](https://doi.org/10.1093/9780197793572.003.0007).

Train and freeze the addition module. Train and freeze a separate comparison module that ranks inputs over `0…10`. Insert a translator from the addition module's output to the comparison module's old left-input representation. Give each possible sum its own urn with eleven old-input choices, initially one each. The right comparison input receives the cardinality of another collection of dots.

The composed task chooses the larger of the sum and that cardinality. On correct comparison reinforce the chosen translator option by one; on failure remove one with floor one. Fixed arithmetic and comparison components produce outputs but receive no updates. The chapter reports mean success of about .94 after one hundred thousand plays in the illustrative transfer setting.

The comparison illustration uses left/right choices; equality needs an explicit policy or exclusion rule. Preserve sum cardinalities as representations that can be mapped incorrectly during transfer. The book's conceptual stages are explicit, while the exact comparison pretraining distribution, tie convention, and selected pretrained checkpoint need clarification for reproduction.

## ASK — A question game

**Source:** [B8 §8.2, pp.131–136, Fig.8.1](https://doi.org/10.1093/9780197793572.003.0008), Barrett and Skyrms (2017).

Nature tells the builder she needs a slab S or a block B with equal probability. The assistant does not see the need. It has one start urn with options Q,S,B and two reply urns keyed by answers A0,A1, each with Q,S,B. The builder has two urns keyed by need after Q, each with A0,A1. All initial weights are one.

If the assistant immediately chooses S or B, the play ends. Correct guesses reinforce the assistant's start decision by two; wrong guesses supply zero. If it asks Q, the builder draws its reply from the need-conditioned urn. The assistant draws from its reply urn and either supplies a material or asks Q again. A successful one-exchange conversation reinforces all its participating draws by 1.5; a failed material choice pays zero.

The source permits repeated Q. Its stated base reward two and one-exchange return 1.5 identify a cost of .5 **per question–reply exchange** for that example. The treatment of arbitrarily long conversations and negative net reward is not fully stated alongside its simple-reinforcement description. A canonical multi-exchange runner therefore needs **Q1** resolved. A one-exchange limit is a useful small variant only when named as such.

There are five urns and thirteen initial weights. The book reports final accuracy above .75 in .534 of ten-million-play runs and above .90 in .398. These final figures differ from cumulative thresholds elsewhere. The interaction structure evolves because the assistant learns whether to initiate discourse and when to act.

## DIAL — Two-question dialogue

**Source:** [B8 §8.3, pp.137–145](https://doi.org/10.1093/9780197793572.003.0008).

Nature uniformly chooses one of four materials: red slab, red block, blue slab, blue block. The assistant can choose any of these actions or ask Q0/Q1. The builder knows the material and can answer A0…A3.

| Module decision | Urn key | Options |
|---|---|---|
| Assistant first move | Start | Four materials, Q0, Q1 |
| Builder reply | Current question × needed material | A0…A3 |
| Assistant second move | First answer | Four materials, Q0, Q1 |
| Assistant third move | Ordered first and second answers | Four materials, Q0, Q1 |

All weights start at one. The assistant's reply keys use **answers**, not a full remembered question–answer history. Builder rows are reused for either question occurrence. A material choice ends play; choosing a third question ends in failure because the builder declines to continue. A correct material reinforces every participating draw by one; failure supplies zero. There are twenty-nine urns: one start, four first-answer rows, sixteen answer-pair rows, and eight builder rows.

The questions and answers are initially meaningless. One question can in principle communicate the full four-way distinction, although reinforcement may produce two complementary questions instead. Record decision occurrences separately when a builder row is reused in a conversation.

**Punishment variant:** add one on success; remove one per chosen occurrence on failure with floor one. The source reports nearly optimal one-question signaling on its simulations.

**Restricted-answer variant:** allow only A0,A1. The assistant then has one start, two first-answer, and four answer-pair urns; the builder retains eight question×material rows. Two questions are required to distinguish all four materials. This variant has fifteen urns.

**Costly variant:** conversations may continue indefinitely; base success payoff three, cost one per question–reply exchange. One correct exchange therefore returns two and a blind guess has expected return .75. The source reports .619 of ten-million-play runs exceeding .90 accuracy. Its unlimited-history storage and handling of failure costs/negative returns need an explicit source profile; use an engineering turn cap only with its configuration label and recorded truncations.

Dialogue requires a sequential interaction driver. A single fixed sender/receiver draw cannot represent the learned choice to ask another question.
