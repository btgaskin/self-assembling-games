import json
import numpy as np
import pytest
from self_assembling_games.toylife.apparatus import (
    resolve_apparatus,
    encode,
    probabilities,
    pack,
    unpack,
)


def test_fieldwise_declaration():
    a = resolve_apparatus(
        {"flag": [False, True], "cue": [False, True], "kind": ["a", "b"]},
        ["a", "b"],
        {},
    )
    assert a == resolve_apparatus(
        {"kind": ["a", "b"], "cue": [False, True], "flag": [False, True]},
        ["a", "b"],
        {},
    )
    molecules = list(a["inputs"].values()) + list(a["targets"].values())
    assert len({min(x, x[::-1]) for x in molecules}) == len(molecules)
    assert a["encoding"]["flag"]["kind"] == "presence"
    assert a["encoding"]["cue"]["kind"] == "one_hot"
    result = encode(a, {"flag": True, "kind": "b"})
    assert result[a["encoding"]["flag"]["channels"][0]] == 1
    assert all(result[c] == 0 for c in a["encoding"]["cue"]["channels"])
    with pytest.raises(ValueError):
        encode(a, {"target": True})
    with pytest.raises(ValueError):
        encode(a, {"kind": "z"})
    with pytest.raises(ValueError):
        encode(a, {"flag": 1})


def test_reverse_collision_and_dose():
    with pytest.raises(ValueError):
        resolve_apparatus(
            {"x": [False, True]},
            ["a", "b"],
            {"ligands": {"input_0": "0001", "act": "1000"}},
        )
    a = resolve_apparatus({"x": [0, 2, 8]}, [0, 1], {"encoding": {"x": "dose"}})
    assert encode(a, {"x": 2}) == {"input_0": 2}
    with pytest.raises(ValueError):
        encode(a, {"x": 2.0})
    with pytest.raises(ValueError):
        resolve_apparatus({"x": [0, 9]}, [0, 1], {"encoding": {"x": "dose"}})


def test_binary_and_batches():
    a = resolve_apparatus({}, [0, 1], {})
    r = {"target_free:act": np.array([2, 0, 3]), "act": np.array([6, 0, 1])}
    np.testing.assert_allclose(
        probabilities(a, r), [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]
    )
    a["output"]["permutation"] = [1, 0]
    np.testing.assert_allclose(probabilities(a, r)[0], [0.75, 0.25])
    np.testing.assert_allclose(probabilities(a, r, [False, True]), [[0, 1]] * 3)
    with pytest.raises(ValueError):
        probabilities(a, r, [False, False])
    with pytest.raises(ValueError):
        probabilities(a, {"target_free:act": 1.0, "act": 2})
    with pytest.raises(ValueError):
        probabilities(a, {"target_free:act": -1, "act": 2})


def test_multi_action_and_pair():
    a = resolve_apparatus({}, ["a", "b", "c"], {})
    r = {"act_0": 2, "act_1": 3, "act_2": 5}
    np.testing.assert_allclose(probabilities(a, r), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(probabilities(a, r, ["a", "c"]), [2 / 7, 0, 5 / 7])
    np.testing.assert_allclose(
        probabilities(a, dict.fromkeys(r, 0), ["a", "c"]), [0.5, 0, 0.5]
    )
    b = resolve_apparatus({}, [0, 1], {"output": "bound_pair"})
    np.testing.assert_allclose(probabilities(b, {"act_0": 2, "act_1": 6}), [0.25, 0.75])
    with pytest.raises(ValueError):
        resolve_apparatus({}, [0, 1], {"output": {"permutation": [0, 0]}})


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype=">i2").reshape(3, 4)[:, ::2],
        np.array(2, dtype="uint64"),
        np.array([], dtype="float32"),
        np.float32(np.nan),
        np.bool_(True),
        np.array(["hi", "world"]),
    ],
)
def test_serialization(array):
    restored = unpack(json.loads(json.dumps(pack({"value": array}))))["value"]
    assert restored.dtype == array.dtype
    assert restored.shape == array.shape
    np.testing.assert_array_equal(restored, array)


def test_serialization_rejects_unsafe_and_corrupt():
    with pytest.raises(ValueError):
        pack(np.array([{}], dtype=object))
    payload = pack(np.arange(3))
    payload["shape"] = [4]
    with pytest.raises(ValueError):
        unpack(payload)


def test_strict_parameters_and_integer_presence():
    for parameters in (
        {"typo": 1},
        {"output": {"typo": 1}},
        {"encoding": {"x": {"kind": "presence", "typo": 1}}},
    ):
        with pytest.raises(ValueError):
            resolve_apparatus({"x": [0, 1]}, [0, 1], parameters)
    a = resolve_apparatus({"x": [0, 1], "cue": [0, 1], "stimulus": [0, 1]}, [0, 1], {})
    assert a["encoding"]["x"]["kind"] == "presence"
    assert a["encoding"]["cue"]["kind"] == "one_hot"
    assert a["encoding"]["stimulus"]["kind"] == "one_hot"
    assert encode(a, {"x": 1})[a["encoding"]["x"]["channels"][0]] == 1
    assert encode(a, {"x": 0})[a["encoding"]["x"]["channels"][0]] == 0
    a = resolve_apparatus({"x": list(range(100))}, [0, 1], {})
    assert all(
        "1" in molecule for molecule in [*a["inputs"].values(), *a["targets"].values()]
    )
