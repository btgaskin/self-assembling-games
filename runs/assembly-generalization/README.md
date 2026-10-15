# Generalization width

Hypothesis: Cross-context reinforcement changes discrimination by coupling learning histories. Compare binary alpha=0,0.1,1 and Gaussian widths 0,0.5,2; alpha=1 provides the exact expected-payoff invariant 1.5.

`experiment.toml` retains the full comparison declaration, seeds and budgets.
These first runs are bounded integration checks. See [study rationale](../../docs/assembly-extension.md) for measures, controls, source relationships and larger-study requirements.

Run with `sg run runs/assembly-generalization/experiment.toml --workspace assembly-generalization`.
