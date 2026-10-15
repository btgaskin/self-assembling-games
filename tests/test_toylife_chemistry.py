"""Optional live CPU chemistry checks against the canonical scalar oracle."""

import json
from collections import Counter

import numpy as np
import pytest

from self_assembling_games.toylife.apparatus import (
    controller,
    encode,
    pack,
    probabilities,
    resolve_apparatus,
    unpack,
)


@pytest.fixture(scope="module")
def native():
    toylife = pytest.importorskip("toylife2")
    from toylife2.specs import ExecutionSpec
    from toylife2.tables import DEFAULT_SOURCE

    if not DEFAULT_SOURCE.is_dir():
        pytest.skip("canonical toyLIFE source tables are unavailable")
    return toylife.Model(execution=ExecutionSpec(backend="cpu"))


def _genes(model):
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 475)[0])
    return np.array([(5 << 16) | coding, 1048465], dtype=np.int32)


def _oracle_output(oracle, genes, state):
    """Capture the oracle's second phase before its canonical boundary release."""
    after_input, _ = oracle.react(state.copy())
    expression = oracle.expression(genes, after_input)
    after_input.proteins = oracle.translated(genes, expression)
    after_input.dimers.clear()
    after_output, _ = oracle.react(after_input)
    return after_output


def _assert_targets(apparatus, readouts, after_output):
    for name, molecule in apparatus["targets"].items():
        expected = sum(
            n for (_, _, _, met), n in after_output.bound.items() if met == molecule
        )
        assert int(readouts[name]) == expected
        assert (
            int(readouts[f"target_free:{name}"]) == after_output.metabolites[molecule]
        )


def test_persistent_fieldwise_supplies_match_canonical_oracle(native):
    from toylife2.chemistry import ChemistrySession
    from toylife2.reference import CellState, ReferenceModel

    apparatus = resolve_apparatus({"cue": [0, 1], "context": [0, 1]}, [0, 1], {})
    spec = controller(apparatus)
    assert spec.mode == "canonical_cell"
    assert all(r.stage == "after_output" for r in spec.ports.readouts)
    genes = _genes(native)
    session = ChemistrySession(native, genes, spec)
    oracle, state = ReferenceModel(native.tables), CellState()
    for cue, context in [(0, 1), (1, 1), (1, 0), (0, 0), (1, 1), (0, 1)]:
        supplies = spec.interface.encode(
            encode(apparatus, {"cue": cue, "context": context})
        )
        for molecule, count in supplies.items():
            state.metabolites[molecule] = int(count)
        after_output = _oracle_output(oracle, genes, state)
        expected = oracle.step(genes, state)
        session.step(supplies)
        assert session.inventories[0].key() == expected.state.key()
        _assert_targets(apparatus, session.last_readouts, after_output)
        state = expected.state


def test_target_occupancy_is_read_before_boundary_release(native):
    from toylife2.chemistry import ChemistrySession

    apparatus = resolve_apparatus(
        {"x": [0, 1]}, [0, 1], {"ligands": {"input_0": "0001", "act": "1111"}}
    )
    spec = controller(apparatus, {"initial_proteins": ((47, 1),)})
    session = ChemistrySession(native, np.array([0, 0], np.int32), spec)
    session.step(spec.interface.encode(encode(apparatus, {"x": 0})))
    assert int(session.last_readouts["act"]) == 1
    assert int(session.last_readouts["target_free:act"]) == 0
    assert not session.inventories[0].bound
    np.testing.assert_array_equal(
        probabilities(apparatus, session.last_readouts), [0, 1]
    )


def test_json_payload_restore_preserves_cue_history_and_readouts(native):
    from toylife2.chemistry import ChemistrySession

    apparatus = resolve_apparatus({"cue": [0, 1]}, [0, 1], {})
    spec, genes = controller(apparatus), _genes(native)
    full = ChemistrySession(native, genes, spec)
    for cue in [0, 1, 1]:
        full.step(spec.interface.encode(encode(apparatus, {"cue": cue})))
    payload = unpack(json.loads(json.dumps(pack(full.payload()))))
    restored = ChemistrySession(native, genes, spec)
    restored.restore_payload(payload)
    assert restored.update == full.update
    assert restored.inventories[0].key() == full.inventories[0].key()
    for cue in [1, 0, 0, 1]:
        supplies = spec.interface.encode(encode(apparatus, {"cue": cue}))
        full.step(supplies)
        restored.step(supplies)
        assert restored.inventories[0].key() == full.inventories[0].key()
        assert restored.update == full.update
        for name, counts in full.last_readouts.items():
            np.testing.assert_array_equal(restored.last_readouts[name], counts)
        np.testing.assert_array_equal(
            probabilities(apparatus, restored.last_readouts),
            probabilities(apparatus, full.last_readouts),
        )


def test_three_action_live_readouts_match_oracle_and_general_decoder(native):
    from toylife2.chemistry import ChemistrySession
    from toylife2.reference import CellState, ReferenceModel

    apparatus = resolve_apparatus({"cue": [0, 1]}, ["left", "stay", "right"], {})
    spec, genes = controller(apparatus), _genes(native)
    session = ChemistrySession(native, genes, spec)
    state, oracle = CellState(proteins=Counter()), ReferenceModel(native.tables)
    for cue in [1, 0, 1]:
        supplies = spec.interface.encode(encode(apparatus, {"cue": cue}))
        for molecule, count in supplies.items():
            state.metabolites[molecule] = int(count)
        after_output = _oracle_output(oracle, genes, state)
        expected = oracle.step(genes, state)
        session.step(supplies)
        _assert_targets(apparatus, session.last_readouts, after_output)
        counts = np.array(
            [int(session.last_readouts[n]) for n in apparatus["output"]["markers"]],
            dtype=float,
        )
        expected_probability = counts / counts.sum() if counts.sum() else np.ones(3) / 3
        np.testing.assert_allclose(
            probabilities(apparatus, session.last_readouts), expected_probability
        )
        masked = probabilities(apparatus, session.last_readouts, ["left", "right"])
        assert masked[1] == 0
        assert masked.sum() == pytest.approx(1)
        assert session.inventories[0].key() == expected.state.key()
        state = expected.state
