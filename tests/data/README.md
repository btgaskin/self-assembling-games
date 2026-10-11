# Domain-refactor behaviour baseline

`domain_refactor_baseline.json` records SHA-256 digests of canonical JSON for
all 85 executable profiles from the Git revision named in that file. It was
captured by running the original checked-in package extracted into a temporary
directory, before comparing the refactored package.

Each case combines the initial snapshot, a final snapshot, and encounter event
records. The fixed seed is 17. Barrett cases advance 24 steps with current-event
recording enabled; finite reinforcement advances 24 plays; evolution advances
0.25 units of evolutionary time. Snapshot metadata is omitted because its source
fingerprint necessarily changes during a code refactor.

The comparison removes only newly introduced descriptive snapshot fields:
`model_specification`, `model_specification_hash`, `model_protocol`, and
`construction_state`. Original configuration, adaptive state, RNG, metrics,
history, compiled-game signatures and events remain in the digest. Canonical
serialization uses `json.dumps(value, sort_keys=True, allow_nan=False)` with
Python's default separators, followed by SHA-256 of the encoded text.

`test_pre_refactor_state_rng_events_and_trajectory` reruns every case. Dedicated
mechanism tests exercise larger controls, feedback routing, structural rollback,
credit failures, shared invocations, evaluation and continuation.
