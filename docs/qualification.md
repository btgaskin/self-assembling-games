# Implementation qualification

Recorded 2026-10-11 on macOS 26.4.1, ARM64, Python 3.14.5, NumPy 2.5.4, pytest 9.1.1, Ruff 0.17.0, setuptools 84.0.0, and build 1.6.1.

The implemented package contains 39 catalogue entries, 32 executable model families, and 45 profiles. All thirteen source chapters remain registered. The catalogue and each profile identify source gaps and deliberate variants.

## Checks performed

- **108 tests passed** in the repository; Ruff checks and formatting checks passed.
- Wheel and source archive built successfully. The wheel includes all four runtime JSON registers; the source archive includes the rulebook and chapter manifest.
- The wheel was installed into a separate temporary target and imported from that target outside the checkout. Model discovery, training, and exact finite evaluation passed.
- Installed-module CLI checks passed for run, inspect, evaluate, resume, replay, and a three-seed ensemble with two independent workers.
- Serial and parallel ensemble equality, every profile's checkpoint continuation, saved-file decision order, learned frozen transfer, source gates, dynamic vocabulary, credit routing, and interaction timing are exercised by the test suite.

These checks qualify the implementation's mechanisms and artifact workflows. The registered million-play and longer book experiments have not been run. Their measured success frequencies remain reference values from the source, rather than results of this implementation.

## Serial timing

Each row is one seed-0 run of 10,000 interactions from the installed wheel, with full event logging disabled. For inquiry an interaction is one consultation. Times include training and metric updates, with simulation construction outside the timed region. These are single measurements on this host.

| Profile | Seconds | Interactions/second | Final tables |
|---|---:|---:|---:|
| `sig.n2.demo` | 0.473 | 21,154 | 4 |
| `sig.n8.punishment.book` | 0.682 | 14,666 | 16 |
| `hier.demo` | 1.084 | 9,228 | 18 |
| `inq.demo` | 0.388 | 25,795 | 5 |

All four runs retained zero complete events. Metrics retain a bounded window; checkpoints retain the current adaptive and world state. Dynamic invention can grow additional tables and has explicit resource limits. Parallel ensembles distribute independent runs; they preserve each seed's serial trajectory.

The measured package source fingerprint is `55517b4565fbe81af34c02a1b9c734c14481d6f01d97b663e375d695fde1bdf5`. Wheel provenance records no repository revision, because the installed artifact lives outside a Git checkout. Source-checkout runs also record their local revision.

## Reproduce

Install the development dependencies in [pyproject.toml](../pyproject.toml), or use the versions in [requirements-qualification.txt](../requirements-qualification.txt). Then run:

```sh
python -m pytest -q
ruff check src tests
ruff format --check src tests
python -m build
sg benchmark sig.n2.demo --plays 10000
sg ensemble sig.n2.demo --seeds 0 1 2 --plays 100 --workers 2 --output runs/check
```

See the [runtime guide](runtime.md) for the experiment protocols and source-reproduction boundaries.
