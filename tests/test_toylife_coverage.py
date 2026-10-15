import json

from self_assembling_games.toylife.coverage import coverage


EXPECTED = set('''SIG COD NAND HIER CUE MAN ORD TR-ORD TR-PART TR-LOG KIND
POLY SCOD COMP-NAND ADD ADD-COMP ASK DIAL DIAL-restricted DIAL-costly VIS STAG
FFSR SUB DISC INQ BOX CHANGE INV HYB BAND META-RULE META-ORACLE MH GEN SET
ATT-R ATT-B RIT SK-SIG SK-COMP SK-VOTE SK-TEAM SK-CHAIN SK-DIALOGUE
ML-COORD ML-TRANSFER ML-COMPOSE ECG'''.split())


def test_every_case_has_explicit_support_and_source_differences():
    rows = coverage()
    assert len(rows) == len(EXPECTED) == 49
    assert {row['id'] for row in rows} == EXPECTED
    statuses = {'native_task', 'derived_task', 'needs_organisation', 'analysis', 'needs_specification'}
    tasks = set('classification boolean context vote ordering addition sum_compare monty_hall delay boxes bandit advisers query conditioning habituation reversal'.split())
    for row in rows:
        assert set(row) == {'id', 'support', 'tasks', 'assumptions', 'requirements'}
        assert row['support'] in statuses
        assert isinstance(row['tasks'], list) and set(row['tasks']) <= tasks
        for field in ('assumptions', 'requirements'):
            assert row[field] and all(isinstance(x, str) and x.strip() for x in row[field])
        # Task support conveys neither historical protocol nor a solved chemistry.
        assert 'source_reproduction' not in row
        assert 'solved' not in row
    assert json.loads(json.dumps(rows)) == rows


def test_coverage_is_detached_and_preserves_organisational_requirements():
    rows = {row['id']: row for row in coverage()}
    for identifier in ('TR-LOG', 'COMP-NAND', 'ML-TRANSFER', 'ML-COMPOSE', 'MAN', 'GEN'):
        assert rows[identifier]['support'] == 'derived_task'
        assert rows[identifier]['requirements']
    assert rows['ECG']['support'] == 'needs_organisation'
    assert rows['DIAL-costly']['support'] == 'needs_specification'
    rows['SIG']['tasks'].append('invented')
    assert 'invented' not in next(row for row in coverage() if row['id'] == 'SIG')['tasks']
