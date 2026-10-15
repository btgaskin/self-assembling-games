# LaCroix contextual logical composition

`LC-LOG` implements two protocols from Travis LaCroix, *Using Logic to Evolve
More Logic: Composing Logical Operators via Self-Assembly*, BJPS 73(2), 407–437
(2022; online 2019), [DOI:10.1093/bjps/axz049](https://doi.org/10.1093/bjps/axz049).
Section and page locators below refer to the
[November 2019 preprint](https://philsci-archive.pitt.edu/16658/7/LaCroix-Evolving-Logic-BJPS-Final-Pre-Print-Nov-2019.pdf).

| Profile | Protocol | Sender observation | Signal relevance |
| --- | --- | --- | --- |
| `lacroix.logic.stipulated` | §3.2, pp15–20, Figures7–8 | A: unary game; B: input | Game→A, input→B, both→both |
| `lacroix.logic.role-free` | §3.5, pp21–22 | Both: game and input | Both send iff context requires both |
| `lacroix.logic.observation-control` | Constructed factorial control | Both: game and input | Retains §3.2 fixed A/B relevance |

Nature independently samples a uniform unary game (`taut` or `neg`), binary
input and context (`game`, `input`, `both`). The executive sender selects A/B/both.
Binary messages select a receiver urn; a missing sender's bit is completed
uniformly. The receiver chooses a symbolic game-input pair. The executive receiver
selects its interpretation from the active sender set. Initially every weight is
one. Successful involved choices gain one; other choices gain zero. All five
functional agents learn from the first encounter.

Success checks the context-relevant symbolic coordinates and signal relevance.
Figure8 requires `(neg,0)` specifically in the `both` context when nature is
`(neg,0)`. `(taut,0)` has the same Boolean output and fails that context.

The source success metric is `success_probability` for a detached policy snapshot;
training summaries also retain cumulative successes. Per-encounter details expose
`context_correct`, `relevance_correct` and `boolean_correct` separately. Boolean
correctness is an auxiliary diagnostic. Unary meanings and their output map are
supplied by these protocols.

The observation control is a declared research case. Section3.4 expands the
receiver's unary action alternatives; it preserves §3.2 sender observations. It
therefore has a different purpose from this observation control. Section3.1's
pretrained comparison and sections3.3/3.4 are outside the implemented profile set.

The configured 10,000 encounters are an execution budget. Published threshold
frequencies and Table9 require a separate benchmark protocol; the preprint table
has a learned-output/full-state column discrepancy relative to its section prose.

```python
from self_assembling_games import model

system = model("lacroix.logic.role-free").instantiate(seed=7)
system.advance(100)
checkpoint = system.snapshot()
```

The example is an execution smoke check. The mechanism tests exercise Figure8,
sender-role permutations, missing-message completion, credit recipients and
checkpoint/trace continuation.
