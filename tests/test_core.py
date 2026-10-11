import numpy as np
import pytest

from self_assembling_games import Address, Feedback, Outcome, Runtime, Table


def test_explicit_credit_fractional_signed_and_frozen():
    rt = Runtime()
    identity = rt.add_table("host", "visitors", ["a", "b"], {"rule": "R1"})
    rt.feedback(Feedback(Address(identity, "b"), True, 0.5))
    assert rt.tables[identity].weights.tolist() == [1, 1.5]
    rt.freeze("host")
    before = rt.tables[identity].to_dict()
    rt.feedback(Feedback(Address(identity, "b"), True, 1))
    assert rt.tables[identity].to_dict() == before
    identity = rt.add_table("executive", "context", [0, 1], {"rule": "SIGNED"})
    rt.feedback(Feedback(Address(identity, 1), True, -0.5))
    assert rt.tables[identity].weights.tolist() == [1, 1]
    assert Outcome(True, 0.5, 1).payoff == -0.5


def test_observation_and_binding_remain_distinct():
    rt = Runtime()
    rt.add_table("sender", 0, [0, 1], {"rule": "R1"})
    rt.record = True
    choice = rt.decide("sender", 0, {"source": 9, "value": 0})
    assert choice.observation == {"source": 9, "value": 0}
    assert choice.table == "sender/0"


def test_r4_literal_vector_and_old_reference():
    rt = Runtime()
    identity = rt.add_table(
        "agent",
        "start",
        [0, 1, 2],
        {
            "rule": "R4",
            "epsilon": 0.2,
            "floor": 0.0001,
            "forgetting": 0.001,
            "w_plus": 0.01,
            "w_minus": 0.02,
        },
    )
    rt.tables[identity].reference = 0.5
    rt.feedback(Feedback(Address(identity, 1), True, 1.0))
    assert rt.tables[identity].weights == pytest.approx([1.099, 1.399, 1.099])
    assert rt.tables[identity].reference == pytest.approx(0.505)
    rt.feedback(Feedback(Address(identity, 0), False, 0.0))
    assert rt.tables[identity].reference == pytest.approx(0.4949)


def test_iterative_phase_uses_play_clock():
    rt = Runtime()
    identity = rt.add_table(
        "a", 0, [0, 1], {"rule": "R3", "reward": 2, "punishment": 4, "phase_length": 2}
    )
    rt.feedback(Feedback(Address(identity, 0), True, 1))
    rt.feedback(Feedback(Address(identity, 0), False, 0))
    assert rt.tables[identity].weights.tolist() == [3, 1]
    rt.clocks["play"] = 2
    rt.feedback(Feedback(Address(identity, 0), True, 1))
    assert rt.tables[identity].weights.tolist() == [3, 1]
    rt.feedback(Feedback(Address(identity, 0), False, 0))
    assert rt.tables[identity].weights.tolist() == [1, 1]


def test_persistent_policy_is_distinct_from_resampling_weights():
    rt = Runtime()
    identity = rt.add_table("a", 0, [0, 1], {"rule": "R6"})
    table = rt.tables[identity]
    table.current = 1
    table.weights[:] = [99, 1]
    table.validate()
    assert table.probabilities().tolist() == [0, 1]
    assert table.probabilities(response=False) == pytest.approx([0.99, 0.01])
    before = table.current
    rt.feedback(Feedback(Address(identity, 1), True, 1))
    assert table.current == before and table.weights[1] == 2
    rt.forced = [0]
    rt.feedback(Feedback(Address(identity, 1), False, 0))
    assert table.current == 0


def test_repeated_occurrences_and_shared_invocation():
    rt = Runtime()
    rt.add_table("a", 0, [0], {"rule": "R1"})
    rt.begin()
    first = rt.decide("a", 0, 0)
    second = rt.decide("a", 0, 0)
    rt.credit([first, second], Outcome(True, 1))
    assert first.occurrence != second.occurrence
    assert rt.tables[first.table].weights[0] == 3
    calls = []
    assert rt.invoke("upstream", lambda: calls.append(1) or 7) == 7
    assert rt.invoke("upstream", lambda: calls.append(2) or 9) == 7
    assert calls == [1]


@pytest.mark.parametrize(
    "weights", [[0, 0], [-1, 1], [float("nan"), 1], [float("inf"), 1]]
)
def test_invalid_urns_rejected(weights):
    with pytest.raises(ValueError):
        Table("a", 0, [0, 1], np.array(weights), {"rule": "R1"})


def test_sampling_returns_ball_and_snapshots_are_isolated():
    rt = Runtime(3)
    identity = rt.add_table("a", 0, [0, 1], {"rule": "R1"})
    before = rt.tables[identity].weights.copy()
    rt.decide("a", 0, 0)
    assert np.array_equal(rt.tables[identity].weights, before)
    state = rt.state()
    state["tables"][identity]["weights"][0] = 99
    assert rt.tables[identity].weights[0] == 1


def test_frozen_module_prevents_new_contexts():
    rt = Runtime()
    rt.freeze("empty_module")
    with pytest.raises(ValueError, match="frozen"):
        rt.add_table("empty_module", 0, [0, 1], {"rule": "R1"})


def test_dynamic_rows_preserve_declared_module_ownership():
    rt = Runtime()
    rt.owners["R"] = "bird"
    rt.module_roles["R"] = "receiver"
    rt.add_table("R", ["a", "b"], [0, 1, 2], {"rule": "R1"})
    assert rt.owners["R"] == "bird" and rt.module_roles["R"] == "receiver"
