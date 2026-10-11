"""Independent operator oracles, participation, and discrete lifecycle checks."""

import itertools
import json

import numpy as np
import pytest

import self_assembling_games as sg
from self_assembling_games.cli import main


def profile(
    model="coord", finite=False, recombination=None, mutation=0.01, initial=None
):
    p = sg.load_adaptive_profile(
        f"ml-{model}.{'wf' if finite else 'deterministic'}.demo"
    )
    for name in p.genetics["transmission"]["mutation"]:
        p.genetics["transmission"]["mutation"][name] = [
            [1 - mutation, mutation],
            [mutation, 1 - mutation],
        ]
    if recombination is not None:
        p.genetics["transmission"]["recombination"] = recombination
    if initial is not None:
        p.protocol["initialization"] = {
            "kind": "explicit_counts" if finite else "frequencies",
            "counts" if finite else "frequencies": initial,
        }
    return p


def independent_recombination(genome, y, definition):
    """Enumerate actual ordered parental pairs and ancestry choices."""
    children = list(map(tuple, genome.coordinates))
    index = {g: i for i, g in enumerate(children)}
    result = np.zeros(genome.count)
    for a, b in itertools.product(range(genome.count), repeat=2):
        pair_mass = y[a] * y[b]
        if definition["kind"] == "clonal":
            result[a] += pair_mass
        elif definition["kind"] == "single_boundary_exchange":
            c = definition["exchange_probability"]
            result[a] += pair_mass * (1 - c)
            for boundary in range(1, len(genome.loci)):
                child = children[a][:boundary] + children[b][boundary:]
                result[index[child]] += pair_mass * c / (len(genome.loci) - 1)
        else:
            for sources in itertools.product((0, 1), repeat=len(genome.loci)):
                mass = 0.5
                for i, r in enumerate(definition["fractions"]):
                    mass *= r if sources[i] != sources[i + 1] else 1 - r
                child = tuple(
                    children[a if source == 0 else b][i]
                    for i, source in enumerate(sources)
                )
                result[index[child]] += pair_mass * mass
    return result


@pytest.mark.parametrize("sizes", [(2, 2), (2, 3, 2)])
@pytest.mark.parametrize(
    "kind", ["clonal", "independent_intervals", "single_boundary_exchange"]
)
def test_transmission_matches_parent_pair_and_mutation_oracles(sizes, kind):
    genome = sg.GenomeSpec(
        [sg.Locus(f"locus{i}", tuple(range(size))) for i, size in enumerate(sizes)]
    )
    rng = np.random.default_rng(14)
    y = rng.dirichlet(np.ones(genome.count))
    mutation = {
        locus.name: rng.dirichlet(
            np.ones(len(locus.alleles)), size=len(locus.alleles)
        ).tolist()
        for locus in genome.loci
    }
    recombination = {"kind": kind}
    if kind == "independent_intervals":
        recombination["fractions"] = [0.13] * (len(sizes) - 1)
    elif kind == "single_boundary_exchange":
        recombination["exchange_probability"] = 0.6
    transmission = sg.TransmissionSpec(
        genome, {"mutation": mutation, "recombination": recombination}
    )
    expected = independent_recombination(genome, y, recombination)
    np.testing.assert_allclose(transmission.recombine(y), expected, atol=1e-15)
    matrix = np.array(
        [
            [
                np.prod(
                    [
                        mutation[locus.name][a[i]][b[i]]
                        for i, locus in enumerate(genome.loci)
                    ]
                )
                for b in genome.coordinates
            ]
            for a in genome.coordinates
        ]
    )
    np.testing.assert_allclose(transmission.mutate(y), y @ matrix, atol=1e-15)
    before, after = genome.allele_frequencies(y), genome.allele_frequencies(expected)
    for name in before:
        np.testing.assert_allclose(before[name], after[name], atol=1e-15)


@pytest.mark.parametrize("fraction", [0, 0.05, 0.5])
def test_standard_two_locus_linkage_and_manuscript_first_generation(fraction):
    p = profile(
        recombination={"kind": "independent_intervals", "fractions": [fraction]}
    )
    simulation = sg.build(p)
    record = simulation.step()
    y, v = (
        np.asarray(record["after_selection"]),
        np.asarray(record["after_recombination"]),
    )

    def D(x):
        return x[0] * x[3] - x[1] * x[2]

    assert D(v) == pytest.approx((1 - fraction) * D(y), abs=1e-15)
    success = np.array([1, 0, 0, 1])
    assert np.asarray(record["expected_offspring"]) @ success == pytest.approx(
        {0: 0.6600666666666666, 0.05: 0.6520633333333332, 0.5: 0.5800333333333334}[
            fraction
        ],
        abs=1e-15,
    )


def test_genotype_association_and_explicit_shared_individual_binding():
    p = profile(mutation=0)
    simulation = sg.build(p)
    for frequencies, expected in (
        ([0.5, 0, 0, 0.5], 1),
        ([0, 0.5, 0.5, 0], 0),
        ([0.25] * 4, 0.5),
    ):
        simulation.set_population(frequencies)
        assert simulation.evaluate()["success_probability"] == expected
        assert simulation.model_instance.realisation.genome.allele_frequencies(
            frequencies
        ) == {"s": [0.5, 0.5], "r": [0.5, 0.5]}
    p.population = {"individuals": {"first": ["sender"], "second": ["receiver"]}}
    independent = sg.build(p)
    independent.set_population([0.5, 0, 0, 0.5])
    assert independent.evaluate()["success_probability"] == 0.5


def test_focal_fitness_integrates_other_individuals_and_zero_weight_contributions():
    p = profile(mutation=0)
    p.population = {
        "individuals": {"first": ["sender"], "second": ["receiver"]},
        "return_weights": {"sender": 1.0, "receiver": 0.0},
    }
    simulation = sg.build(p)
    x = np.array([0.4, 0.1, 0.2, 0.3])
    simulation.set_population(x)
    game = simulation.game
    expression = simulation.model_instance.realisation.expression.plan_indices
    expected = np.zeros(4)
    for focal in range(4):
        for other in range(4):
            policies = {
                "sender": np.eye(4)[expression["sender"][focal]],
                "receiver": np.eye(4)[expression["receiver"][other]],
            }
            expected[focal] += (
                x[other] * game.expectations(policies)["expected_returns"]["sender"]
            )
    np.testing.assert_allclose(
        simulation.evaluate()["reproductive_weights"], 1 + expected
    )


@pytest.mark.parametrize("shared", [True, False])
def test_matrix_payoffs_and_complete_plan_costs_follow_individual_binding(shared):
    p = profile(mutation=0)
    p.model = "SK-SIG"
    p.parameters = {"cost_case": "listening", "cost_interpretation": "table_literal"}
    p.genetics["expression"] = {
        "plan_indices": {"sender": [1, 1, 2, 2], "receiver": [1, 2, 1, 2]}
    }
    p.population = {
        "individuals": {"cell": ["sender", "receiver"]}
        if shared
        else {"a": ["sender"], "b": ["receiver"]}
    }
    simulation = sg.build(p)
    simulation.set_population([0.4, 0.1, 0.2, 0.3])
    x = simulation.frequencies
    expression = simulation.model_instance.realisation.expression.plan_indices
    expected = np.zeros(4)
    for i in range(4):
        if shared:
            policies = {s: np.eye(4)[expression[s][i]] for s in simulation.game.roles}
            returns = simulation.game.expectations(policies)["expected_returns"]
            expected[i] = 0.5 * sum(returns.values())
        else:
            for j in range(4):
                for focal_slot in simulation.game.roles:
                    policies = {
                        s: np.eye(4)[expression[s][i if s == focal_slot else j]]
                        for s in simulation.game.roles
                    }
                    expected[i] += (
                        0.5
                        * x[j]
                        * simulation.game.expectations(policies)["expected_returns"][
                            focal_slot
                        ]
                    )
    np.testing.assert_allclose(
        simulation.evaluate()["reproductive_weights"], 1 + expected
    )


def test_composition_stage_oracles_and_clonal_functional_reduction():
    even = [0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0]
    pure = [1, 0, 0, 0, 0, 0, 0, 0]
    success = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    for initial, after_recombination, after_mutation in (
        (even, 0.75, 0.735298),
        (pure, 1.0, 0.970596),
    ):
        simulation = sg.build(
            profile(
                "compose",
                initial=initial,
                recombination={
                    "kind": "single_boundary_exchange",
                    "exchange_probability": 0.5,
                },
            )
        )
        record = simulation.step()
        assert np.asarray(record["after_recombination"]) @ success == pytest.approx(
            after_recombination
        )
        assert np.asarray(record["expected_offspring"]) @ success == pytest.approx(
            after_mutation
        )
    beta = 3 * 0.01 * 0.99**2 + 0.01**3
    expected = 1.0
    first, second = (sg.build(profile("compose", initial=x)) for x in (even, pure))
    for _ in range(200):
        selected = 2 * expected / (1 + expected)
        expected = selected * (1 - beta) + (1 - selected) * beta
        first.step()
        second.step()
        assert first.evaluate()["success_probability"] == pytest.approx(
            expected, abs=1e-14
        )
        assert second.evaluate()["success_probability"] == pytest.approx(
            expected, abs=1e-14
        )
    assert expected == pytest.approx(0.9429703176710604)


def test_equal_expression_retains_different_transmission_neighbourhoods():
    p = profile(mutation=0.01, initial=[1, 0, 0, 0])
    # 00 and 01 express the same complete phenotype; their routes to 10 differ.
    p.genetics["expression"]["plan_indices"] = {
        "sender": [1, 1, 2, 2],
        "receiver": [2, 2, 2, 1],
    }
    first = sg.build(p)
    p.protocol["initialization"]["frequencies"] = [0, 1, 0, 0]
    second = sg.build(p)
    assert (
        first.evaluate()["success_probability"]
        == second.evaluate()["success_probability"]
    )
    assert first.step()["expected_offspring"][2] == pytest.approx(0.0099)
    assert second.step()["expected_offspring"][2] == pytest.approx(0.0001)


@pytest.mark.parametrize("target", [[1, 0], [1, 1]])
def test_template_encoding_reverses_target_accessibility(target):
    p = profile("transfer", initial=[1, 0, 0, 0])
    p.parameters = {"target": target}
    first = sg.build(p)
    p.genetics["expression"]["plan_indices"]["adapter2"] = [0, 1, 1, 0]
    second = sg.build(p)
    assert (
        first.evaluate()["success_probability"]
        == second.evaluate()["success_probability"]
        == 0.5
    )
    a, b = first.step()["expected_offspring"], second.step()["expected_offspring"]
    target_separable = 2 * target[0] + target[1]
    target_coupled = 2 * target[0] + (target[0] ^ target[1])
    assert (a[target_separable] > b[target_coupled]) == (target == [1, 0])


def test_entire_transmission_coordinate_transport_is_a_null():
    genome = sg.GenomeSpec([sg.Locus("a", (0, 1)), sg.Locus("b", (0, 1))])
    transmission = sg.TransmissionSpec(
        genome,
        {
            "mutation": {
                "a": [[0.99, 0.01], [0.01, 0.99]],
                "b": [[0.99, 0.01], [0.01, 0.99]],
            },
            "recombination": {"kind": "independent_intervals", "fractions": [0.5]},
        },
    )
    permutation = np.array([0, 1, 3, 2])
    original = np.array([0.4, 0.3, 0.2, 0.1])
    relabelled = original[permutation]
    fitness = np.array([1, 1, 2, 1])
    for _ in range(50):
        selected = original * fitness / (original @ fitness)
        original = transmission.mutate(transmission.recombine(selected))
        renamed_selection = (
            relabelled * fitness[permutation] / (relabelled @ fitness[permutation])
        )
        # Conjugate the entire nonlinear transmission operation.
        relabelled = transmission.mutate(
            transmission.recombine(renamed_selection[permutation])
        )[permutation]
        np.testing.assert_allclose(original[permutation], relabelled, atol=1e-15)
    # WF transition means/covariances transport too; seeded category paths need not.
    np.testing.assert_allclose(
        np.diag(original[permutation])
        - np.outer(original[permutation], original[permutation]),
        (np.diag(original) - np.outer(original, original))[permutation][:, permutation],
    )


def test_wright_fisher_conditional_mean_covariance_and_count_authority():
    simulation = sg.build(profile(finite=True, initial=[50] * 4), seed=42)
    r = simulation.model_instance.realisation
    initial = r.counts.copy()
    expected = sg.build(profile()).step()["expected_offspring"]
    draws = []
    for _ in range(6000):
        r.counts = initial.copy()
        r.advance()
        assert sum(r.counts) == 200
        draws.append(r.frequencies)
    draws = np.array(draws)
    expected = np.array(expected)
    covariance = (np.diag(expected) - np.outer(expected, expected)) / 200
    np.testing.assert_allclose(draws.mean(axis=0), expected, atol=0.0015)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False, ddof=0), covariance, atol=0.00006
    )
    exposed = r.frequencies
    exposed[:] = 0
    assert sum(r.frequencies) == pytest.approx(1)


@pytest.mark.parametrize("finite", [False, True])
def test_snapshot_continuation_inspection_and_evaluation_preserve_state(
    finite, tmp_path
):
    simulation = sg.build(
        profile(finite=finite), seed=19, protocol={"sample_interval": 3}
    )
    assert simulation.run(0)["generation_average_expected_success"] is None
    simulation.run(4)
    snapshot = simulation.snapshot()
    inspection = simulation.model_instance.inspect()
    inspection["realisation"]["life_cycle"].append("external_edit")
    assert simulation.snapshot() == snapshot
    other = sg.build(profile(finite=finite), seed=19)
    assert "external_edit" not in other.fidelity["adaptive_law"]["life_cycle"]
    assert "external_edit" not in simulation.model_instance.inspect()["realisation"][
        "life_cycle"
    ]
    sg.evaluate(simulation)
    assert simulation.snapshot() == snapshot
    simulation.checkpoint(tmp_path)
    restored = sg.resume(tmp_path)
    assert restored.snapshot() == snapshot
    assert simulation.step() == restored.step()
    assert restored.snapshot() == simulation.snapshot()
    assert [m["generation"] for m in simulation.measurements] == [0, 3]
    assert simulation.summary()["generation_average_expected_success"] == pytest.approx(
        simulation.success_sum / 5
    )


def test_generation_failure_is_atomic_including_rng_and_stage_record(monkeypatch):
    simulation = sg.build(profile(finite=True), seed=8)
    simulation.step()
    before = simulation.snapshot()
    r = simulation.model_instance.realisation
    expected = r.expectations
    calls = 0

    def failure(values=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Failed next-census assay")
        return expected(values)

    monkeypatch.setattr(r, "expectations", failure)
    with pytest.raises(ValueError, match="next-census"):
        simulation.step()
    assert simulation.snapshot() == before


@pytest.mark.parametrize("finite", [False, True])
def test_generation_average_counts_unsampled_generations(finite):
    a = sg.build(profile(finite=finite), seed=9, protocol={"sample_interval": 1})
    b = sg.build(profile(finite=finite), seed=9, protocol={"sample_interval": 7})
    a.run(10)
    b.run(10)
    assert a.summary() == b.summary()
    np.testing.assert_array_equal(a.frequencies, b.frequencies)


@pytest.mark.parametrize("method", ["haplotype.deterministic", "wright_fisher.haploid"])
def test_cli_generation_clock_resume_and_parallel_ensemble(method, tmp_path, capsys):
    p = profile(finite=method == "wright_fisher.haploid")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(p.to_dict()))
    root = tmp_path / "run"
    assert main(["run", str(path), "--generations", "4", "--output", str(root)]) == 0
    capsys.readouterr()
    assert main(["resume", str(root), "--generations", "3"]) == 0
    capsys.readouterr()
    assert sg.resume(root).generation == 7
    assert main(["evaluate", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == 7
    assert (
        main(["run", str(path), "--time", "1", "--output", str(tmp_path / "wrong")])
        == 2
    )
    capsys.readouterr()
    assert sg.benchmark(p, generations=2)["clock"] == "generations"
    serial = sg.run_ensemble(p, [4, 9], generations=3, output=tmp_path / "serial")
    parallel = sg.run_ensemble(
        p, [4, 9], generations=3, workers=2, output=tmp_path / "parallel"
    )
    assert serial["runs"] == parallel["runs"]
    assert (
        sg.run_ensemble(p, [4, 9], generations=3, output=tmp_path / "serial") == serial
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.genetics.update(population_size=10), "population_size"),
        (lambda p: p.population.update(rates={"population": 1}), "rates"),
        (lambda p: p.protocol.update(mutation_rate=0.01), "Protocol fields"),
        (lambda p: p.protocol.update(generations=1.5), "generations"),
        (lambda p: p.protocol.update(sample_interval=0.5), "sample_interval"),
        (
            lambda p: p.population.update(individuals={"cell": ["sender", "sender"]}),
            "partition",
        ),
        (
            lambda p: p.genetics["expression"]["plan_indices"].update(sender=[0, 1]),
            "plan index",
        ),
        (
            lambda p: p.genetics["transmission"].update(
                recombination={"kind": "independent_intervals", "fractions": [0.6]}
            ),
            "fraction",
        ),
        (lambda p: p.genetics["fitness"].update(background=-1), "Fitness"),
        (lambda p: p.resources.update(max_haplotypes=2), "max_haplotypes"),
        (lambda p: p.resources.update(max_genotype_mask_entries=2), "mask_entries"),
        (lambda p: p.resources.update(max_transmission_work=1), "work budget"),
        (lambda p: p.resources.update(max_field_evaluations=2), "resource budgets"),
    ],
)
def test_shared_resolution_gates(mutate, match):
    p = profile()
    mutate(p)
    with pytest.raises((ValueError, TypeError), match=match):
        sg.ModelSpecification.from_profile(p)


def test_signed_game_returns_use_explicit_reproductive_transformation():
    p = profile()
    p.model = "SK-SIG"
    p.parameters = {"background": -1}
    p.genetics["fitness"] = {"background": 2, "scale": 1}
    simulation = sg.build(p)
    assert min(simulation.evaluate()["reproductive_weights"]) >= 0
    simulation.step()
    p.genetics["fitness"] = {"background": 0, "scale": 1}
    with pytest.raises(ValueError, match="Transformed reproductive"):
        sg.build(p)


def test_zero_mean_reproduction_and_resource_failures_preserve_state():
    p = profile()
    p.genetics["fitness"] = {"background": 0, "scale": 0}
    with pytest.raises(ValueError, match="Mean reproduction"):
        sg.build(p)
    p = profile()
    p.resources = {"max_measurements": 1}
    simulation = sg.build(p)
    before = simulation.snapshot()
    with pytest.raises(ValueError, match="max_measurements"):
        simulation.step()
    assert simulation.snapshot() == before


def test_old_methods_reject_genetics_and_checkpoint_binding_tampering():
    p = profile()
    p.method = "replicator.continuous"
    with pytest.raises(ValueError, match="Genetics belongs"):
        sg.build(p)
    snapshot = sg.build(profile(finite=True)).snapshot()
    snapshot["counts"][0] += 1
    with pytest.raises(ValueError, match="Census counts"):
        sg.restore_adaptive(snapshot)
    snapshot = sg.build(profile()).snapshot()
    snapshot["model_specification"]["realisation"]["parameters"]["expression"][
        "plan_indices"
    ]["sender"][0] = 0
    with pytest.raises(ValueError, match="specification hash"):
        sg.restore_adaptive(snapshot)


def test_declared_allele_order_and_single_locus_exchange_gate():
    genome = sg.GenomeSpec([sg.Locus("z", ("second", "first")), sg.Locus("a", (1, 0))])
    assert genome.haplotypes == [
        ["second", 1],
        ["second", 0],
        ["first", 1],
        ["first", 0],
    ]
    with pytest.raises(ValueError, match="internal boundary"):
        sg.TransmissionSpec(
            sg.GenomeSpec([sg.Locus("a", (0, 1))]),
            {
                "recombination": {
                    "kind": "single_boundary_exchange",
                    "exchange_probability": 0.5,
                }
            },
        )


def test_empty_initializer_and_work_history_budgets_are_explicit():
    p = profile()
    p.protocol["initialization"] = {}
    with pytest.raises(ValueError, match="Initialization"):
        sg.build(p)
    p = profile()
    p.resources["max_generation_work"] = 1
    with pytest.raises(ValueError, match="max_generation_work"):
        sg.ModelSpecification.from_profile(p)
    p = profile()
    p.resources["max_history_entries"] = 1
    simulation = sg.build(p)
    before = simulation.snapshot()
    with pytest.raises(ValueError, match="max_history_entries"):
        simulation.run(2)
    with pytest.raises(ValueError, match="max_history_entries"):
        simulation.set_population([1, 0, 0, 0])
    assert simulation.snapshot() == before


def test_native_specification_overrides_and_custom_factory_use_standard_methods(
    tmp_path,
):
    import copy

    specification = sg.ModelSpecification.from_profile(profile())
    game = specification.construct_finite_game()
    override = copy.deepcopy(game)
    override.fidelity["game"]["questions"].append("ML-OVERRIDE-Q")
    simulation = sg.build_adaptive(specification, game=override)
    simulation.run(3)
    assert (
        "ML-OVERRIDE-Q"
        in simulation.model_instance.specification.to_dict()["source"]["questions"]
    )
    assert (
        sg.restore_adaptive(simulation.snapshot(), game=override).snapshot()
        == simulation.snapshot()
    )
    identifier = "multilocus-custom-fixture"

    def factory(parameters, fixed_decisions, **budgets):
        def encounter(provider):
            act = provider.decide("individual", "act")
            return sg.ParticipantOutcome(act == 1, (float(act == 1),))

        return sg.FiniteGame(
            identifier,
            ("individual",),
            (sg.FiniteDecision("individual", "act", (0, 1), "effector"),),
            encounter,
            parameters,
            {"game": {"status": "explicit_constructed_assay", "questions": []}},
            fixed_decisions,
            **budgets,
        )

    sg.register_finite_game(
        identifier,
        factory,
        family="test",
        description="One locus, one effector",
        source="constructed:custom-fixture",
    )
    spec = sg.ModelSpecification.resolve(
        game=identifier,
        game_parameters={},
        realisation="population.multilocus",
        realisation_parameters={
            "genome": {"loci": [{"name": "gene", "alleles": [0, 1]}]},
            "expression": {"plan_indices": {"individual": [0, 1]}},
        },
    )
    custom = sg.build(spec)
    assert custom.step()["after_census"] == pytest.approx([1 / 3, 2 / 3])
    assert sg.restore_adaptive(custom.snapshot()).snapshot() == custom.snapshot()
