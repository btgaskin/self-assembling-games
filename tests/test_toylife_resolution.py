"""Pure declaration checks run without importing the optional chemistry backend."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from self_assembling_games.config import from_dict, load
from self_assembling_games.domain import model


TASKS = 'classification boolean context vote ordering addition sum_compare monty_hall delay boxes bandit advisers query conditioning habituation reversal'.split()
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('task', TASKS)
def test_task_presets_resolve_and_roundtrip(task):
    resolved = model(f'toylife.{task}')
    data = resolved.to_dict()
    assert data['interaction']['game'] == task
    assert data['organisation']['owner'] == 'cell'
    assert data['source']['kind'] == 'deliberate_variant'
    assert from_dict(data).to_dict() == data


@pytest.mark.parametrize('name,table', [
    ('nand', [1, 1, 1, 0]), ('and', [0, 0, 0, 1]),
    ('or', [0, 1, 1, 1]), ('xor', [0, 1, 1, 0]),
])
def test_boolean_aliases_and_overrides(name, table):
    resolved = model(f'toylife.boolean.{name}').to_dict()
    assert resolved['interaction']['parameters']['truth_table'] == table
    assert from_dict(resolved).to_dict() == resolved
    altered = model(f'toylife.boolean.{name}', interaction={'parameters': {'truth_table': [0, 0, 0, 0]}})
    assert altered.to_dict()['interaction']['parameters']['truth_table'] == [0, 0, 0, 0]


@pytest.mark.parametrize('section,parameters', [
    ('interaction', {'unknown': 1}), ('realisation', {'unknown': 1}),
    ('adaptation', {'unknown': 1}), ('execution', {'unknown': 1}),
    ('realisation', {'genes': True}), ('adaptation', {'mutation_probability': 2}),
    ('execution', {'response_updates': 0}),
])
def test_unknown_and_invalid_parameters_are_rejected(section, parameters):
    with pytest.raises(ValueError):
        model('toylife.boolean', **{section: {'parameters': parameters}})


@pytest.mark.parametrize('name', ['capacity', 'experience', 'apparatus', 'transfer'])
def test_smoke_workspaces_resolve(name):
    resolved = load(ROOT / 'runs' / f'toylife-{name}' / 'experiment.toml')
    data = resolved.to_dict()
    assert data['workers'] == 1
    assert data['seeds'] == [0]
    assert all(c['budget'] == 2 for c in data['conditions'])
    assert all(c['model']['runtime'] == 'self_assembling_games.toylife' for c in data['conditions'])
    assert from_dict(data).to_dict() == data


def test_resolution_does_not_import_optional_toylife_dependency():
    script = '''
import importlib.abc, json, sys
class BlockChemistry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('toylife', 'toylife2') or fullname.startswith(('toylife.', 'toylife2.')):
            raise AssertionError('Resolution imported optional chemistry: ' + fullname)
sys.meta_path.insert(0, BlockChemistry())
from self_assembling_games.domain import model
model('toylife.boolean.nand')
print(json.dumps([name for name in sys.modules if name in ('toylife', 'toylife2') or name.startswith(('toylife.', 'toylife2.'))]))
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True, env=env)
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize('arguments,field', [
    (['list', '--family', 'toylife'], None),
    (['catalogue'], 'presets'),
])
def test_cli_discovers_molecular_presets_without_chemistry(arguments, field):
    script = '''
import importlib.abc, json, sys
class BlockChemistry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'toylife2' or fullname.startswith('toylife2.'):
            raise AssertionError('Discovery imported optional chemistry')
sys.meta_path.insert(0, BlockChemistry())
from self_assembling_games.cli import main
main(json.loads(sys.argv[1]))
'''
    result = subprocess.run([sys.executable, '-c', script, json.dumps(arguments)],
                            check=True, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')))
    rows = json.loads(result.stdout)
    if field:
        rows = rows[field]
    identifiers = {row['id'] for row in rows}
    assert {f'toylife.{task}' for task in TASKS} <= identifiers
    assert {f'toylife.boolean.{name}' for name in ('nand', 'and', 'or', 'xor')} <= identifiers
