import copy
import json

import pytest

from self_assembling_games import System, build, model
from self_assembling_games.component_artifacts import (
    export_component,
    load_component_snapshot,
)
from self_assembling_games.finite_artifacts import import_frozen_decisions
from self_assembling_games.simulation import digest


def transfer(preset, templates):
    return model(
        preset,
        interaction={
            "parameters": {"template_mode": "artifact", "templates": templates}
        },
    ).instantiate(seed=13)


def test_public_system_checkpoint_export_transfer_and_detached_evaluation(tmp_path):
    source = model("nand.demo").instantiate(seed=17)
    source.advance(25)
    before = source.snapshot()
    checkpoint = source.checkpoint(tmp_path / "source.json")
    path = export_component(checkpoint, tmp_path / "component")
    artifact = json.loads(path.read_text())
    assert source.snapshot() == before
    assert artifact["source"]["checkpoint_hash"] == before["checkpoint_hash"]
    assert artifact["source"]["model_hash"] == before["model_hash"]
    assert artifact["source"]["position"] == 25
    assert artifact["source"]["seed"] == 17
    assert artifact["source"]["declaration"] == before["model"]["source"]
    assert artifact["evaluation"]["success_probability"] == pytest.approx(
        source.evaluate()["success_probability"]
    )
    checkpoint.unlink()
    target = transfer("tr-log.demo", {"template": str(path)})
    state = target.snapshot()
    lineage = state["state"]["lineage"]["template"]
    assert lineage["artifact_hash"] == artifact["artifact_hash"]
    assert lineage["component"]["source"]["checkpoint_hash"] == before["checkpoint_hash"]
    frozen = {
        k: v for k, v in state["state"]["runtime"]["tables"].items() if v["frozen"]
    }
    target.advance(12)
    assert frozen == {
        k: v for k, v in target.snapshot()["state"]["runtime"]["tables"].items()
        if v["frozen"]
    }
    saved = target.snapshot()
    path.unlink()
    restored = System.restore(saved)
    restored.advance(7)
    target.advance(7)
    assert restored.snapshot() == target.snapshot()


@pytest.mark.parametrize("envelope", ["legacy", "system"])
def test_direct_checkpoint_import_preserves_order(envelope, tmp_path):
    source = model("nand.demo").instantiate(seed=3)
    source.advance(8)
    checkpoint = source.snapshot()
    if envelope == "legacy":
        checkpoint = checkpoint["state"]
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint, sort_keys=True))
    target = transfer("tr-log.demo", {"template": str(path)})
    order = target.snapshot()["state"]["runtime"]["table_order"]
    assert [k.removeprefix("template:") for k in order if k.startswith("template:")] == source.snapshot()["state"]["runtime"]["table_order"]


def test_same_artifact_installs_independent_frozen_components(tmp_path):
    source = model("nand.demo").instantiate(seed=11)
    source.advance(9)
    path = export_component(source, tmp_path / "component.json")
    target = transfer("comp-nand.2.demo", {"first": str(path), "second": str(path)})
    runtime = target._backend.simulation.runtime
    first, second = runtime.tables["first:A/0"], runtime.tables["second:A/0"]
    assert first.weights.tolist() == second.weights.tolist()
    first.weights[0] += 5
    first.validate()
    assert first.weights.tolist() != second.weights.tolist()
    assert load_component_snapshot(path)["runtime"]["tables"]["A/0"]["weights"] == second.weights.tolist()


def test_export_preserves_assignments_error_and_copies_complete_policy(tmp_path):
    source = build("nand.demo", seed=9)
    source.set_policy("A", 0, 1)
    source.runtime.tables["A/0"].learner["error"] = 0.2
    source.runtime.tables["A/0"].reference = 3.5
    before = source.snapshot()
    path = export_component(source, tmp_path)
    loaded = load_component_snapshot(path)
    expected = copy.deepcopy(before["runtime"]["tables"]["A/0"])
    expected["frozen"] = True
    assert loaded["runtime"]["tables"]["A/0"] == expected
    assert source.snapshot() == before
    loaded["runtime"]["tables"]["A/0"]["options"].append(99)
    assert load_component_snapshot(path)["runtime"]["tables"]["A/0"] == expected
    imported = import_frozen_decisions(path, {"sender/state:0": "A/0"})
    assert imported["fixed_decisions"]["sender/state:0"] == pytest.approx([0.1, 0.9])


def test_artifact_integrity_and_immutable_destination(tmp_path):
    source = model("nand.demo").instantiate()
    path = export_component(source, tmp_path)
    assert export_component(source, tmp_path) == path
    source.advance(1)
    with pytest.raises(FileExistsError, match="already exists"):
        export_component(source, path)
    artifact = json.loads(path.read_text())
    artifact["policy"]["tables"]["A/0"]["weights"][0] += 1
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="integrity"):
        load_component_snapshot(path)
    with pytest.raises(ValueError, match="integrity"):
        transfer("tr-log.demo", {"template": str(path)})


def test_explicit_subcomponent_selection_and_interface_validation(tmp_path):
    source = model("comp-nand.2.demo").instantiate()
    with pytest.raises(ValueError, match="explicit NAND"):
        export_component(source, tmp_path / "implicit")
    specification = {
        "interface": "NAND",
        "modules": {m: "first:" + m for m in ["A", "B", "R"]},
    }
    path = export_component(source, tmp_path / "first", specification)
    artifact = json.loads(path.read_text())
    assert artifact["evaluation"]["success_probability"] == 1
    assert artifact["transformations"][1]["bindings"] == specification["modules"]
    assert all(not row["module"].startswith("first:") for row in artifact["policy"]["tables"].values())
    invalid = copy.deepcopy(specification)
    invalid["modules"]["R"] = "P"
    with pytest.raises(ValueError, match="pair rows|receiver row|options"):
        export_component(source, tmp_path / "bad", invalid)


@pytest.mark.parametrize("preset", ["ord.demo", "add.demo"])
def test_invented_vocabularies_require_transforms_and_are_reusable(preset, tmp_path):
    source = model(preset).instantiate(seed=3)
    source.advance(100)
    before = source.snapshot()
    with pytest.raises(ValueError, match="explicit mutator"):
        export_component(source, tmp_path / "implicit")
    path = export_component(source, tmp_path / "explicit", {
        "suppress_mutator": True, "complete_pair_rows": True,
    })
    artifact = json.loads(path.read_text())
    assert source.snapshot() == before
    assert len(artifact["policy"]["table_order"]) == len(artifact["policy"]["tables"])
    assert all("mutator" not in row["options"] for row in artifact["policy"]["tables"].values())
    assert 0 <= artifact["evaluation"]["success_probability"] <= 1
    if preset == "ord.demo":
        transfer("tr-ord.demo", {"ordering": str(path)}).advance(4)
    else:
        assert load_component_snapshot(path)["profile"]["model"] == "ADD"


def test_missing_pair_requires_declared_completion(tmp_path):
    source = build("nand.demo")
    source.runtime.remove(source.runtime.table_id("R", [0, 1]))
    with pytest.raises(ValueError, match="Unseen pair"):
        export_component(source, tmp_path / "implicit")
    path = export_component(source, tmp_path / "explicit", {"complete_pair_rows": True})
    artifact = json.loads(path.read_text())
    assert artifact["transformations"][-1]["added_rows"] == 1


def test_complete_plan_import_is_never_marginalized(tmp_path):
    from self_assembling_games import build_adaptive

    source = build_adaptive("sk-sig.strategy-r1.demo", seed=4)
    checkpoint = source.checkpoint(tmp_path)
    with pytest.raises(ValueError, match="complete-plan correlations"):
        import_frozen_decisions(checkpoint, {"sender/state:0": "sender/state:0"})
    with pytest.raises(ValueError, match="correlated complete plans"):
        export_component(checkpoint, tmp_path / "component")


def test_component_rejects_invalid_order_even_with_valid_content_hash(tmp_path):
    path = export_component(model("nand.demo").instantiate(), tmp_path)
    artifact = json.loads(path.read_text())
    artifact["policy"]["table_order"] = artifact["policy"]["table_order"][:-1]
    artifact.pop("artifact_hash")
    artifact["artifact_hash"] = digest(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="every table once"):
        load_component_snapshot(path)
