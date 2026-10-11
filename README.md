# Self-assembling games

A source-based specification for Jeffrey A. Barrett's reinforcement-learning models of signaling, rule transfer, modular composition, discourse, networks, inquiry, and learning how to learn.

This is a separate research project from toyLIFE. It contains the rulebook and an executable reinforcement runtime with a Python API, CLI, exact/sampled evaluation, frozen composition, dynamic vocabulary, dialogue, networks, inquiry, and nested learning. Source-specific operational questions remain attached to the affected profiles.

## Run a game

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/sg list
.venv/bin/sg run sig.n2.demo --seed 0 --plays 10000 --output runs/signaling
.venv/bin/sg evaluate runs/signaling
.venv/bin/python -m pytest -q
```

```python
from self_assembling_games import build, evaluate

simulation = build("hier.demo", seed=0)
simulation.run(plays=10_000)
state = simulation.inspect()
assessment = evaluate(simulation.snapshot())
```

See the [qualification report](docs/qualification.md) for measured checks and timings, and the [runtime guide](docs/runtime.md) for profiles, training artifacts, CLI commands, and reproducibility. [Astra review](docs/astra-review.md) and the [repository plan](docs/repository-plan.md) explain the structural decisions.

## Read the structure

1. [Catalogue](docs/catalogue.md): the model families across all thirteen chapters, with implementation readiness and boundaries.
2. [Architecture](docs/architecture.md): conditional urns, observations, decisions, feedback, timing, and composition.
3. [Learning rules](docs/learning-rules.md): the precise updates that the games use.
4. Game specifications: [basic and generalized signaling](docs/games/signaling.md), [assembly and transfer](docs/games/transfer.md), [computation and discourse](docs/games/computation-discourse.md), [networks and inquiry](docs/games/networks-inquiry.md), and [adaptive learning](docs/games/adaptive-learning.md).
5. [Source register](docs/sources.md): chapter titles, printed pages, DOI links, quotations, and cited antecedents.
6. [Review](docs/review.md): implementation decisions and outstanding source questions.

The primary source is Barrett, Jeffrey A. (2025), *Self-Assembling Games: Language, Learning, and Inquiry*, Oxford University Press. Chapter citations use **B1–B13**, section numbers, and the book's printed page numbers. The source register links every chapter. PDF fingerprints and Zotero identifiers are in [sources/chapters.json](sources/chapters.json).

## What an urn means

An urn stores a weight for each available option. Drawing means sampling an option with probability equal to its weight divided by the total weight. Updating the weight changes the chance of choosing that option in the same circumstances later. Floating-point weights represent fractional balls without creating physical ball objects.

The sampled draws make a run stochastic; random nature and randomized scheduling can add further randomness. The weights themselves are explicit state. Given the initial state, configuration, and recorded random draws, the trajectory can be replayed. A deterministic expected-payoff calculation or mean-field model is a separate execution mode.

Barrett discusses the relationship between simple reinforcement and replicator dynamics in **B2, pp.25–27**. This first specification focuses on his reinforcement models. Population reproduction, replicator integration, and Bayesian updating can subsequently use the same game definitions with their own state and update semantics.

## Fidelity policy

Each experiment will name its source model, configuration, and any deliberate variation. A small demonstration can shorten a run or shrink a configurable game while retaining its mechanism. A reproduction uses the source parameters and the source's measurement definition. A qualitative proposal in the book stays a proposal until its remaining operational choices have been made explicit.

The proposed common abstraction is an implementation design derived from the models. Barrett does not present an axiomatized mathematical category with specified objects and morphisms. Here, “category” means the organized family of games and their compositional operations. [Architecture](docs/architecture.md) sets out those operations without attributing additional formal claims to the book.
