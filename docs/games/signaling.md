# Basic and generalized signaling

All tables below specify conditional acts. Initial weight is one per listed option unless stated otherwise. “Reward the trace” means update the selected option for each participating decision occurrence. See [learning rules](../learning-rules.md).

## SIG — Lewis–Skyrms signaling

**Source:** [B2 §2.3, pp.27–34](https://doi.org/10.1093/9780197793572.003.0002), David Lewis (1969), Brian Skyrms (2010).

Nature chooses state `x` from `n` states uniformly. The sender observes `x`; the receiver observes only the signal. Each of the sender's `n` urns contains `n` signals. Each of the receiver's `n` urns contains `n` matching actions. Draw signal from sender urn `x`, then action from receiver urn `signal`. If `action=x`, reward both selected decisions by one; otherwise update neither. Sampling leaves the original weights in place.

| Module | Contexts | Options | Urns |
|---|---|---|---|
| Sender | State | Signal | `n` |
| Receiver | Signal | Action | `n` |

The canonical profile has `2n²` weights. A general `i×j×k` game uses `i` sender rows of `j` signals and `j` receiver rows of `k` actions; specify a success/payoff matrix when states and actions no longer match by index. Biased nature is a named profile with explicit state probabilities.

For `n=2` and uniform nature, the source reports almost-sure convergence to perfect signaling in the limit, citing Argiento et al. (2009). Larger or biased games can pool states. A uniformly distributed `n=3` example succeeds about two thirds of the time; four-state pooling often succeeds about three quarters. These are outcomes of learned policies, not additional action types.

**Reference experiments:** `n=4`, one million plays, about .76 of runs exceed .95 **cumulative** accuracy. For `n=8`, one hundred million plays, about .40 exceed .95 cumulative accuracy. Keep the denominator and threshold explicit. The chapter's approximately .21 long-run failure figure for four states uses a different endpoint from the million-play .76 success figure.

**Variants:** R2 punishment, biased nature, and the whole-strategy win-stay/lose-randomize comparator. The eight-state `(+1,−1)` example exceeds .95 cumulative accuracy in about .83 of ten-million-play runs. These are separate experiment configurations.

## COD — Coding game

**Source:** [B3 §3.1, pp.39–41](https://doi.org/10.1093/9780197793572.003.0003), Barrett (2007).

Nature chooses one of four states uniformly. Both senders A and B see the full state, each has four urns containing binary signals `0,1`, and neither observes the other's current draw. The receiver has four ordered-pair urns `00,01,10,11`, each containing four actions. Draw both signals, form `(A_signal,B_signal)`, and draw an action. Reward all three draws by one on a match; failure pays zero.

There are twelve urns and thirty-two weights. A perfect convention partitions the four states into two classes per sender whose intersections distinguish every state. The partition emerges; initial sender observations remain full-state observations throughout. It is possible to learn partial information or pooling instead.

**Derived example:** A codes states `{0,1}` as zero and `{2,3}` as one; B codes `{1,2}` as zero and `{0,3}` as one. Then `01→0`, `00→1`, `10→2`, `11→3`. This repairs the printed example's reference to state “4” where the domain is `0…3`; it is an explicitly corrected illustration, not a new learning constraint.

The book reports .73 of million-play runs exceeding .80 cumulative accuracy. Punishment is discussed as an improvement. A primitive compositional expression here always contains both sender terms; the hierarchical game below also uses terms individually.

## NAND — Evolving signaling and a logical function

**Source:** [B3 §3.2, pp.41–43](https://doi.org/10.1093/9780197793572.003.0003), related models in Skyrms (2004,2010).

Nature supplies two independent uniform truth values `SA,SB`. Sender A sees only `SA` and B sees only `SB`. Each has two urns with signals green/red. The receiver has four ordered signal-pair urns, each with acts T/F. Success means F on `(T,T)`, T on all other truth-input pairs. On success reward A, B, and receiver by one; failure pays zero.

| `SA` | `SB` | Required act |
|---|---|---|
| F | F | T |
| F | T | T |
| T | F | T |
| T | T | F |

There are eight urns and sixteen weights. Signals can invert their eventual truth-value conventions. Always acting T earns .75 with uniform inputs, so raw accuracy must be compared with that baseline. The book reports, for one thousand million-play runs, fractions .71/.62/.50 exceeding cumulative thresholds .80/.90/.95 respectively.

The same structure can target any of the sixteen binary truth functions. That is a parameterized capability grounded in the chapter's discussion, not a claim of equal convergence probabilities. NAND's functional completeness supports serial composition; compatible learned representations still matter at the module boundary.

## HIER — Hierarchical composition

**Source:** [B3 §3.3, pp.43–53, Fig.3.3](https://doi.org/10.1093/9780197793572.003.0003), Barrett, Skyrms and Cochran (2018); Barrett et al. (2020).

Nature independently chooses a property pair `black/white × dog/cat` and a context `color/animal/both`, all uniformly. A and B each see the complete property pair; executive sender ES sees only context. Executive receiver ER sees which senders signaled. The basic receiver R sees the expression.

| Module | Urn keys | Options | Number |
|---|---|---|---|
| A | Four property pairs | `0A,1A` | 4 |
| B | Four property pairs | `0B,1B` | 4 |
| ES | Color, animal, both | Activate A, B, or both | 3 |
| ER | A, B, or both sent | Interpret color, animal, or both | 3 |
| R | Ordered binary pair | Four property-pair actions | 4 |

Execution:

1. ES selects the active sender set from the context urn.
2. Only active basic senders draw their terms. A one-term expression preserves sender identity.
3. If both terms arrive, R uses their pair urn. If one arrives, choose the missing binary value uniformly, thereby selecting one of the two compatible receiver urns. This is a chance step, not another learned choice.
4. ER selects interpretation from its urn for the active set.
5. R's drawn property pair is projected onto color, animal, or both according to ER. Success requires the selected interpretation to match the requested context and its projected property to match nature.
6. Compute `π = (1.5 if a simple-context success; 2.0 if both-context success; 0 otherwise) − .5 × number_of_terms`. Add `π` to each used decision's selected weight, flooring at one.

There are eighteen urns, fifty weights, and thirty-two independent probability coordinates before learning. These counts are derived from the table. The uniform missing-bit completion changes no learned table by itself. Negative reward reaches ES, ER, R, and active basic senders; inactive senders receive no feedback.

For an efficient perfect convention, one sender represents color and the other animal. ES sends just the relevant term in simple contexts and both in the combined context; ER learns the corresponding output type. Sender assignments and bit polarities may exchange. The minimum average term count is `4/3`, and canonical net reward is one in every successful context. These are derived optima, useful for checking implementation.

The book reports near-perfect efficient signaling on .975 of one thousand runs of one hundred million plays, typically high efficiency within one million. Equal base payoffs of two with cost .5 also work well; lower costs .3 and .1 reduce the reported rate of efficient composition. Keep success and efficiency as separate measures.

Suboptimal conventions can include a color sender and a second sender encoding a crosscutting, XOR-like property. Such terms help in combination while failing to identify an animal alone. The model therefore needs unrestricted full-state sender tables even when the desired optimum is the simple color/animal division.
