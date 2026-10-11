# Astra document review

Astra (`gpt-6-astra`) reviewed every tracked file at documentation commit `1ea74a1`: twelve Markdown documents, the chapter manifest, and `.gitignore`. The review used those documents and made no repository changes or external-source requests.

## Structural findings adopted

1. Feedback targets adaptive state independently of sampled choices. Reciprocal host learning, new-token credit, ARP, and generalization require this.
2. Permitted observations and policy-table bindings are separate. ATT-B supports explicitly named source/value and shared-value variants while its source binding remains open.
3. Chance events and structural transitions supplement decision occurrences for replay.
4. Success and numerical return are separate. A successful HIER act can incur negative net payoff.
5. A module invocation evaluates once; every consumer reuses its output. Shared and copied modules have explicit instance identities.
6. R1–R7 label source mechanisms of different kinds: numeric updates, schedules, invention, persistent response, and nested tasks.
7. Mechanism readiness, executable-profile completeness, source-reproduction completeness, implementation, and validation have separate meanings.
8. Learned ordering precedes learned ordering transfer. Constructed templates are useful explicit fixtures for testing adapters.

The review found the documented finite urn/weight counts internally consistent. It also derived the NAND input-transfer closure: ten functions under arbitrary unary recodings, four under bijective recodings, fourteen with output complementation, with XOR/XNOR outside that family. The runtime tests these results.

## Core and implementation direction

The required core consists of scoped state, stable identities, permitted observations, policy bindings, response policies, chance events, explicit outcomes and feedback addresses, timing, structural transactions, experiment lifecycle, and inspection. These contracts support readable game-specific functions and reusable composition helpers.

Astra recommended a serial Python/NumPy reference implementation, bounded active-interaction records, optional streamed event logs, independent evaluation randomness, reproducible frozen artifacts, and source-gated profiles. Independent-run parallelism follows deterministic qualification.

## Review boundary

This was an internal document review. Earlier book-reading provenance is retained in the source register; original-paper questions remain attached to individual profiles. The current runtime was subsequently checked through the test suite described in [runtime.md](runtime.md).
