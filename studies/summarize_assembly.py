"""Render the retained smoke report without running or changing any simulation."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'studies/assembly-smoke-results.json').read_text())
lines = ['# Assembly extension: retained smoke studies', '',
         'Seven declared comparison groups exercise acquired-state release, component reuse,',
         'joint hierarchical learning, cross-context reinforcement and divergent utilities.',
         'Each condition uses seeds 0, 1 and 2. All endpoint evaluations below are exact.',
         'These short runs measure early dispositions and check integration; their small seed',
         'set and horizon support exploratory comparisons.', '',
         'The [machine-readable report](assembly-smoke-results.json) retains each endpoint,',
         'verification result, run path, source hash and elapsed time. The [runner](assembly_smoke.py)',
         'uses the archived measurements, and every run retains its original and resolved TOML,',
         'checkpoint states, receipts and numerical outputs. The [interpretation](assembly-interpretation.md)',
         'connects these results and explains the additional release control. A [compressed run archive](assembly-smoke-runs.tar.gz)',
         'contains the full retained directories; its SHA-256 is recorded in the JSON report.', '',
         '| Study | Trials | Wall time (s) | Verified |', '|---|---:|---:|---|']
for study in data['studies']:
    name = Path(study['configuration']).parent.name
    lines.append(f"| {name} | {len(study['evaluation']['trials'])} | {study['elapsed_seconds']:.2f} | {study['verification']['valid']} |")
lines += ['', f"Peak resident memory across the reporting processes: {data['peak_process_rss_bytes']/1024**2:.1f} MiB.",
          'Timing includes running, retaining and verifying the study. The composed cases include',
          'exact enumeration of both stochastic components at each measurement.', '',
          '## Endpoint comparisons', '',
          'Success is the declared task predicate. For generalization it means exact action/state',
          'agreement; Gaussian reward also credits nearby actions. Conflict utilities are reported',
          'separately. ODE comparisons end at time 1; learning comparisons use their declared encounter',
          'budgets. The range is across three seeds, not a confidence interval.', '']
for study in data['studies']:
    name = Path(study['configuration']).parent.name
    lines += [f'### {name}', '', f"[Configuration](../{study['configuration']}) · [Retained run](../{study['run']})", '',
              '| Condition | Mean success | Seed range |', '|---|---:|---:|']
    groups = defaultdict(list)
    for trial in study['evaluation']['trials']:
        evaluation = trial['phases'][-1]['evaluation'] if 'phases' in trial else trial['evaluation']
        groups[trial['condition']].append(evaluation)
    for condition, values in groups.items():
        scores = [v['success_probability'] for v in values]
        lines.append(f'| {condition} | {mean(scores):.4f} | {min(scores):.4f}–{max(scores):.4f} |')
    lines += ['']
    if name == 'assembly-generalization':
        for condition, values in groups.items():
            lines.append(f"- {condition}: mean expected reward {mean(v['expected_net_payoff'] for v in values):.4f}; normalised reward {mean(v['normalized_expected_payoff'] for v in values):.4f}.")
    if name == 'assembly-conflict':
        for condition, values in groups.items():
            returns = [v.get('expected_owner_returns',v.get('expected_returns')) for v in values]
            formatted = ', '.join(f'{owner}={mean(r[owner] for r in returns):.4f}' for owner in sorted(returns[0]))
            lines.append(f'- {condition}: mean participant utilities {formatted}.')
    lines += ['']
(root / 'studies/assembly-smoke.md').write_text('\n'.join(lines))
