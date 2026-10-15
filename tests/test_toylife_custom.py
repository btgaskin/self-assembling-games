"""Generic interaction contracts with a deterministic chemistry test double."""

import importlib
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from self_assembling_games.toylife import interaction, resolve_model
from self_assembling_games.toylife.apparatus import encode
from self_assembling_games.toylife.runtime import AssayEvaluator


@pytest.fixture
def custom_model(tmp_path, monkeypatch):
    module_name = "temporary_chemical_interaction"
    path = tmp_path / f"{module_name}.py"
    path.write_text(
        textwrap.dedent("""
        from self_assembling_games.situated import Stage, StageResult
        class Custom:
            actors = ('cell',)
            fields = {'cue': [0, 1]}
            actions = ['quiet', 'respond']
            def __init__(self, parameters): self.parameters = parameters
            def initial_world(self, rng):
                return {'phase': 0, 'secret': 987654, 'ledger': []}
            def stage(self, world):
                i = world['phase']
                return Stage('custom', ('cell',) if i == 2 else (),
                             duration=[2, self.parameters.get('rest', 3), 0][i])
            def observations(self, world):
                return {'cell': {'cue': 1} if world['phase'] == 0 else {}}
            def legal_actions(self, world, actor): return self.actions
            def transition(self, world, actions, rng):
                i = world['phase']
                world['ledger'].append({'phase': i, 'actions': actions})
                world['phase'] += 1
                if i < 2:
                    assert not actions
                    return StageResult(world, info={'private_result': world['secret']})
                assert [row['phase'] for row in world['ledger']] == [0, 1, 2]
                return StageResult(world, feedback={'cell': {'score': float(actions['cell'] == 'respond')}},
                                   terminated=True, info={'ledger': world['ledger']})
        def factory(parameters): return Custom(parameters)
    """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    import sys

    sys.modules.pop(module_name, None)
    data = resolve_model("toylife.delay", {})
    data["interaction"] = {"provider": module_name + ":factory", "parameters": {}}
    data["source"] = {"metadata_secret": 123456}
    data["realisation"]["parameters"]["apparatus"] = {}
    data["realisation"]["parameters"].pop("resolved_apparatus")
    data["execution"]["parameters"].update(episodes=1, response_updates=7)
    return data


@pytest.fixture
def fake_chemistry(monkeypatch):
    pytest.importorskip("toylife2")
    import toylife2.chemistry
    import self_assembling_games.toylife.runtime as runtime

    class ChemicalMemory:
        sessions = []

        def __init__(self, model, genes, controller):
            self.update = 0
            self.history = []
            self.retained_cue = False
            self.last_readouts = {"act": np.int32(0), "target_free:act": np.int32(1)}
            self.sessions.append(self)

        def step(self, supplied):
            self.history.append(dict(supplied))
            self.retained_cue |= any(supplied.values())
            self.update += 1
            self.last_readouts = {
                "act": np.int32(self.retained_cue),
                "target_free:act": np.int32(not self.retained_cue),
            }

    monkeypatch.setattr(toylife2.chemistry, "ChemistrySession", ChemicalMemory)
    monkeypatch.setattr(
        runtime,
        "controller",
        lambda apparatus, parameters: SimpleNamespace(
            interface=SimpleNamespace(encode=lambda supplied: supplied)
        ),
    )
    return ChemicalMemory


def test_custom_resolves_without_builtin_game_and_keeps_source_inspectable(
    custom_model,
):
    resolved = resolve_model(custom_model, {})
    custom = interaction(resolved)
    assert custom.fields == {"cue": [0, 1]}
    assert custom.actions == ["quiet", "respond"]
    assert importlib.import_module(type(custom).__module__).__file__.endswith(".py")


@pytest.mark.parametrize("rest, expected_updates", [(3, 5), (None, 9)])
def test_custom_timing_private_world_and_persistent_chemistry(
    custom_model, fake_chemistry, rest, expected_updates
):
    custom_model["interaction"]["parameters"]["rest"] = rest
    data = resolve_model(custom_model, {})
    data["source"]["metadata_secret"] = 123456
    native_model = SimpleNamespace(
        engine=SimpleNamespace(_check_memory=lambda *args, **kwargs: None)
    )
    evaluator = AssayEvaluator(native_model, data, seed=4)
    result = evaluator.evaluate(np.array([[0, 0]], dtype=np.int32))
    assert result.scores[0] == 1
    assert evaluator.metrics == {"chemical_updates": expected_updates, "world_ticks": 3}
    assert len(fake_chemistry.sessions) == 1
    session = fake_chemistry.sessions[0]
    panel = data["realisation"]["parameters"]["resolved_apparatus"]
    assert session.history[:2] == [encode(panel, {"cue": 1})] * 2
    assert session.history[2:] == [encode(panel, {})] * (expected_updates - 2)
    assert all(set(supply) == set(panel["inputs"]) for supply in session.history)
    assert [row["observation"] for row in evaluator.diagnostics] == [{"cue": 1}, {}, {}]
    assert evaluator.diagnostics[-1]["actions"] == {"cell": "respond"}
    ledger = evaluator.diagnostics[-1]["info"]["ledger"]
    assert [row["phase"] for row in ledger] == [0, 1, 2]
    # Nonacting phase info and source metadata never become chemical input.
    assert evaluator.diagnostics[0]["info"] == {"private_result": 987654}
    assert session.retained_cue
