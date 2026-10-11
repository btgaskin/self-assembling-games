# Assembly, rule following, and template transfer

## CUE / MAN — Cue reading and sensory manipulation

**Source:** [B4 §§4.1–4.3, especially pp.63–68](https://doi.org/10.1093/9780197793572.003.0004).

These are one-sided precursors to mutual signaling. Cue reading fixes the association from natural states to observable sender behavior, while a receiver learns the successful response. Sensory manipulation fixes a receiver's signal-to-action dispositions while a sender learns how to trigger the appropriate action. In the simple two-state illustrations the fixed map is deterministic, matching utility is one, and nonmatching utility is zero.

Represent the fixed side as a module with a specified policy and updates disabled; represent the adaptive side with its conditional urns and success reinforcement. The observations and fixed map belong in the source profile. If both sides later receive fitness-relevant benefits and learn, a signaling relationship may develop. The book gives the ritualization account; an exact switch time, transition schedule, and complete observation-reinforcement algorithm are additional experiment choices.

## ORD — Bird ordering with invented signals

**Source:** [B5 §5.4, pp.79–82](https://doi.org/10.1093/9780197793572.003.0005), Barrett and Skyrms' ordering model discussed there.

Two senders each observe one of seven ranked stimuli. Independent uniform sampling chooses the two current stimuli. Each sender has seven input urns initially containing only a black mutator. Drawing it proposes a fresh sender-specific signal; drawing an existing token sends it. The receiver observes the ordered pair of tokens, creates a pair urn as needed, and draws one of `<,>,=` with initial weight one each.

Success means the receiver's ordering relation matches the rank relation of the two stimuli. Existing successful choices gain one in each used urn; failures gain zero. A successful newly invented signal is introduced at weight one into every input urn of its sender and reinforced in its origin urn. Retaining signals only after successful first use permits a learned vocabulary without preassigning names to ranks.

The two sender identities remain meaningful even when token strings resemble one another. Dynamic receiver rows are keyed by their ordered, typed pair. The chapter describes receiver urn creation and successful vocabulary retention; first-failure receiver-row disposal needs checking against the underlying ordering implementation. Track it as question **Q3** in [review](../review.md).

## TR-ORD — Transfer a learned ordering

**Source:** [B5 §5.5, pp.82–85](https://doi.org/10.1093/9780197793572.003.0005).

First train the ordering game on the old seven stimuli. Freeze both old senders and the receiver. Give each new stimulus a translator urn whose options are the seven old stimulus/input-urn identities, initially weight one. Each current new stimulus selects an old input urn, from which the fixed sender produces a signal; the fixed receiver interprets the pair.

Nature samples the two new stimuli freely and without bias. Each sender has its own seven-row translator table. Reward the translator choices by one when the relation is correct; punish those choices by one on failure, with a floor of one. Old sender and receiver draws remain stochastic according to their frozen learned policies, and their weights remain fixed. Translator state is the new learned component; it does not relabel or overwrite the old game.

## TR-PART — Partial training and inference tests

**Source:** [B5 §5.6, pp.85–87](https://doi.org/10.1093/9780197793572.003.0005).

Train the new ordering relation on adjacent stimulus pairs, then test nonadjacent comparisons. Preserve the source's direction and sampling convention when constructing adjacent training pairs. Evaluation records predictions without learning updates. Separate training accuracy from held-out relation accuracy and from transfer accuracy over the entire domain.

The book's incomplete-information discussion shows how a learned template can constrain responses beyond directly reinforced comparisons. An experimentally biased or incorrect old template is a distinct intervention, not a generic assumption that transfer discovers the correct order.

## TR-LOG — Transfer a logical operation

**Source:** [B5 §5.8, pp.89–92](https://doi.org/10.1093/9780197793572.003.0005).

Pretrain a NAND game, then freeze its two sender modules and receiver. Two translators each see a new binary input and choose an old T/F input urn. Each translator therefore has two urns with two options, initialized uniformly. Pass those selected inputs through the frozen game. Reward translator choices by one if the output matches the target operation on the new inputs; failure pays zero.

For NAND transfer the correct translators preserve input truth values up to learned representation conventions. For OR transfer, inversion of both inputs makes `NAND(¬R,¬S)=R∨S`. Compare transfer with scratch learning under the same target distribution, inner response rule, and accuracy measure. Count pretraining cost separately when interpreting total efficiency.

The chapter suggests transferring other truth functions using a small set of templates. Independent input translators and a fixed output do not automatically reach every function sharing the same number of false truth-table rows. For example, projections and XOR have two false rows but differ in whether both inputs matter. Any broad equivalence claim needs an explicit allowed transformation set and exhaustive verification; see **Q5**.

## KIND — Learned concepts and equivalence classes

**Source:** [B5 §5.7, pp.87–89](https://doi.org/10.1093/9780197793572.003.0005).

Learned response patterns can individuate conventional kinds or concepts. Treat this as an analysis of learned policies: determine which stimuli induce the same distributions or successful classifications under a declared criterion. The discussion does not supply a new fully parameterized urn game. Thresholding approximate similarity is an analysis choice that should accompany its output.

## POLY — Polymerization

**Source:** [B6 §6.1, pp.98–101](https://doi.org/10.1093/9780197793572.003.0006).

Serial chains of successful interactions can become a larger interaction. Represent the candidate component games, legal output-to-input connections, and overall feedback explicitly. Barrett describes this assembly mechanism qualitatively; a particular chain's transition timing and reward allocation must be specified before simulation.

## SCOD — Self-assembling coding through salience

**Source:** [B6 §§6.2–6.3, pp.101–110](https://doi.org/10.1093/9780197793572.003.0006), Barrett (2023a).

Use the coding game's four-state nature and two binary senders. Add a receiver salience urn with options `A`, `B`, and `A&B`, initially one each. The receiver has eight action urns: two for A's single terms, two for B's single terms, and four for ordered pairs. Each action urn contains the four possible acts once.

On a play both senders draw and send. The receiver draws its attention option, observes only the selected sender(s), then draws an act from the corresponding action urn. A correct act adds one to both sender decisions, the receiver's salience choice, and its action choice. Failure changes nothing. Receiver attention does not deactivate a sender; this is different from HIER's executive activation.

There are seventeen urns and fifty-one weights. The receiver learns that a pair can provide information unavailable from one term, while both sender partitions and all receiver actions coevolve. Successful composition selects the informative joint source; it is learned through the same kind of conditional decision as an ordinary signal.
