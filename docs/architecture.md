# Models, experiments and runs

A **Model** declares the mechanisms of an interaction. An **Experiment** resolves comparison conditions, independent seeds, budgets and measurements. A **Run** records an execution of that experiment. Python and TOML resolve to the same declarations before any participants are constructed or random choices are drawn.

## Four scientific components

| Component | What it determines | Examples of consequential choices |
|---|---|---|
| Interaction | Available observations and actions, their order, world transitions, outcomes and feedback | Private colours; simultaneous movement; a supplied signal alphabet; observation noise; reward recipients |
| Organisation | Which components belong to which individuals and how they meet | Shared genotype across roles; independent individuals sampled from one population; changing adviser assignments |
| Realisation | How stored structure produces behaviour, including intrinsic controller state | Conditional propensities; complete inherited plans; genotype-to-plan expression; retained movement history |
| Adaptation | How experience or reproduction changes that structure | Reinforcement credit; reference points; mutation; recombination; selection and census sampling |

Execution supplies a clock and a budget. Resource limits constrain optional computations such as exact strategy enumeration. Source declarations record the provenance, assumptions and unresolved choices of a particular case.

These components are configuration boundaries. Providers implement compatible combinations and reject unsupported combinations. The package does not create a separate class for every scientific distinction. A custom runtime is an importable module implementing `resolve_model`, `create_backend` and `restore_backend`; the backend implements `advance`, `snapshot`, `evaluate`, `inspect` and clock, position and status properties. The [situated protocol](situated-interactions.md) provides a smaller extension point for new worlds and controllers.

## Material state and adaptive dispositions

A bound **System** holds the authoritative running state. A policy reports what a participant currently tends to do. Its underlying state also determines how it can change: proportional urns can express the same action probabilities while responding differently to the next reward. Complete plans preserve correlations between decisions. Multilocus distributions preserve differences between inherited structures even when those structures express the same behaviour.

The realisation therefore retains weights, assignments, correlations, controller memory and inherited structure. Adaptation acts on that state. Inspection exposes the configured mechanism and its live organisation; evaluation derives measurements from a detached copy. Normalised policies are useful observations of the system, rather than sufficient checkpoints.

Intrinsic world and controller transitions occur during interaction. Adaptive feedback has its own recipients and timing. Performance, credited reward and reproductive fitness have separate meanings: a costly successful action can receive negative net payoff, and a constructed reproductive mapping can transform payoff before selection.

The built-in Barrett kernels retain their source-defined encounter boundaries. Nested META tasks create and finish their inner learners within one outer encounter; their checkpoint boundary is between complete outer tasks. Situated systems additionally expose `step_stage()` for mid-round checkpoints. Each provider describes its actual boundary.

## How signalhood enters a model

Source signalling games supply roles, signal alphabets and observation channels. Those are declared features of their interactions. Their adaptation explains how conventions develop within those conditions.

The ECG-inspired fixture supplies a world with movement, waiting and finishing. A participant sees its own colours and both positions. The same movements can accomplish travel and become available for communicative use. There is no privileged `send` action or stored signal meaning. The fixture's random and scripted controllers exercise the world protocol; studies of acquired signalhood require an additional adaptive controller and operational measurements.

A movement's informative correlation, a receiver's causal sensitivity to it, and a mutually maintained convention are different measurements. Investigating their relation calls for interventions on access, timing, pairing, history and adaptive feedback. A field named `meaning` or `common_ground` would supply the explanation that such an experiment is intended to investigate.

## Resolution and persistence

`model()` resolves defaults and validates declarations. `instantiate()` constructs seeded state. Exact compilation is an explicit capability; sampled conditional execution and situated interactions can operate without enumerating complete plans. Resource limits apply when an exact calculation is requested.

Snapshots include the resolved model, state, random generator state, protocol, configuration hashes and implementation identity. Restoration checks their integrity and provenance. Situated extensions also fingerprint imported interaction, controller and adaptation modules. Checkpoints support continuation within the pinned implementation; migration of running state across package versions requires a separate migration protocol.

Experiments archive resolved inputs and per-trial checkpoints in a timestamped study directory. Each new execution receives a new directory. Resumption loads the archived declaration. See the [experiment workflow](experiments.md) for TOML precedence, files and CLI commands.

## Extension sequence

New experiments should first vary existing mechanisms through their declared parameters. A new interaction can use the situated protocol with the same lifecycle and run storage. A new realisation needs its own sufficient state and a compatible adaptation mechanism. Tests should compare uninterrupted and restored execution, verify observation boundaries, and show that inspection and evaluation preserve running state.

The optional toyLIFE2 provider exposes sequence-to-structure-to-function through realisation, molecular/world transitions through interaction, and inherited variation through adaptation. Its declared apparatus maps observations to ligands and molecular readouts to actions. One organism owns one shared chemical pool; chemical state persists within episodes and organism-level selection acts between generations. See the [adapter guide](toylife.md) for task coverage, timing, fitness and evaluation controls. Additional compartments and continuous chemistry require their own explicit organisational and dynamical contracts.

## Comparisons supported by this structure

A useful first study varies nature's prior in SIG while retaining the same learning rule, initial weights and encounter budget; `runs/signalling/experiment.toml` supplies a small starting declaration. Record both expected endpoint success and its trajectory across independent pairs. Threshold time is measured on retained samples, with unreached trials censored at their budget.

Further controlled comparisons can distinguish:

- Conditional reinforcement and complete-plan reinforcement on the same finite interaction, measuring the consequences of correlations between decisions.
- Independent role populations and shared inheritance, with individual bindings stated separately from population membership.
- Fixed and adaptive components in transfer tasks, using held-out inputs and recording exactly which decisions receive feedback.
- Deterministic haplotype recursions and Wright–Fisher samples, with population size and generation clock explicit.
- Alternative reward, cost and reproductive-fitness mappings while retaining the interaction's success criterion.
- Retained controller history and reset history in situated interactions, followed by partner swaps or observation interventions when a suitable learning controller is implemented.

Cross-method comparisons retain native clocks: encounters, evolutionary time and generations have distinct meanings. Estimate uncertainty across independent learner pairs or population replicates. Within-run events describe trajectories and do not create additional independent replicates. The ECG workspace currently compares supplied controllers to exercise the protocol; acquired signalling, dialogue and common-ground hypotheses require a declared learning mechanism and corresponding causal interventions.
