"""Run declared assembly smoke comparisons and retain a machine-readable report.

Usage: PYTHONPATH=src python studies/assembly_smoke.py
Each invocation creates fresh run directories. Configs determine all budgets/seeds.
"""
from pathlib import Path
import json
import resource
import sys
import time

from self_assembling_games import run, System
from self_assembling_games.conflict_profiles import signalling_information
from self_assembling_games.simulation import provenance


def main():
    root = Path(__file__).resolve().parents[1]
    configs = sorted(root.glob('runs/assembly-*/experiment.toml'))
    configs += sorted(root.glob('runs/assembly-lifecycle/*/experiment.toml'))
    configs = sorted(set(configs))
    report = {'provenance': provenance(), 'studies': []}
    failed = False
    for config in configs:
        started = time.perf_counter()
        result = run(str(config), directory=root / 'runs', workspace=str(config.parent.relative_to(root / "runs")))
        status = result.status()
        verification = result.verify()
        verification["path"] = str(result.path.relative_to(root))
        # Use the already retained endpoint measurement instead of rerunning it.
        measurements = []
        for trial in status['trials']:
            trial_path = result.path / trial['path']
            measured = {'condition': trial['condition'], 'seed': trial['seed']}
            if status['schema_version'] == 3:
                measured['phases'] = []
                for receipt in sorted(trial_path.glob('phases/*/results.json')):
                    phase = json.loads(receipt.read_text())
                    measured['phases'].append({'id': phase['id'], 'position': phase['position'],
                                              'evaluation': phase['samples'][-1]['evaluation']})
            else:
                measured['evaluation'] = json.loads((trial_path / 'results.json').read_text())['evaluation']
            measurements.append(measured)
        evaluation = {'trials': measurements}
        if status['schema_version'] == 2:
            for trial, measured in zip(status['trials'], evaluation['trials']):
                checkpoint = result.path / trial['path'] / 'checkpoint.json'
                if checkpoint.exists():
                    system = System.restore(json.loads(checkpoint.read_text()))
                    if system.model.to_dict()['interaction']['game'] == 'SIG':
                        measured['signalling_information'] = signalling_information(system)
        elapsed = time.perf_counter() - started
        entry = {'configuration':str(config.relative_to(root)),
                 'run':str(result.path.relative_to(root)),
                 'status': status['status'], 'verification': verification,
                 'evaluation': evaluation, 'elapsed_seconds':elapsed}
        report['studies'].append(entry)
        failed |= status['status'] != 'complete' or not verification['valid']
        print(f"{config.parent.name}: {status['status']} ({elapsed:.2f}s)", flush=True)
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_process_rss_bytes'] = memory if sys.platform == 'darwin' else memory * 1024
    report['note'] = 'Bounded integration studies; peak RSS covers this sequential process, not an isolated trial.'
    output = root / 'studies' / 'assembly-smoke-results.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if failed:
        raise SystemExit('A study failed; inspect retained manifests.')


if __name__ == '__main__':
    main()
