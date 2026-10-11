"""Small multilocus comparisons; run with PYTHONPATH=src python this_file.py."""

from self_assembling_games import build, load_adaptive_profile


def coordination():
    for fraction in (0, 0.05, 0.5):
        profile = load_adaptive_profile("ml-coord.deterministic.demo")
        profile.genetics["transmission"]["recombination"] = {
            "kind": "independent_intervals",
            "fractions": [fraction],
        }
        simulation = build(profile)
        simulation.step()
        print(
            "Coordination, r =",
            fraction,
            "after one generation:",
            simulation.evaluate()["success_probability"],
        )


def composition():
    even = [0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0]
    profile = load_adaptive_profile("ml-compose.deterministic.demo")
    profile.protocol["initialization"] = {"kind": "frequencies", "frequencies": even}
    profile.genetics["transmission"]["recombination"] = {
        "kind": "single_boundary_exchange",
        "exchange_probability": 0.5,
    }
    record = build(profile).step()
    success = [1, 0, 0, 1, 0, 1, 1, 0]
    for stage in (
        "before",
        "after_selection",
        "after_recombination",
        "expected_offspring",
    ):
        print("Composition,", stage, sum(p * z for p, z in zip(record[stage], success)))


def transfer():
    for target in ([1, 0], [1, 1]):
        for encoding in ("separable", "coupled"):
            profile = load_adaptive_profile("ml-transfer.deterministic.demo")
            profile.parameters["target"] = target
            profile.protocol["initialization"] = {
                "kind": "frequencies",
                "frequencies": [1, 0, 0, 0],
            }
            if encoding == "coupled":
                profile.genetics["expression"]["plan_indices"]["adapter2"] = [
                    0,
                    1,
                    1,
                    0,
                ]
            simulation = build(profile)
            target_index = 2 * target[0] + (
                target[1] if encoding == "separable" else target[0] ^ target[1]
            )
            first = simulation.step()
            print(
                "Target",
                target,
                encoding,
                "target offspring probability:",
                first["expected_offspring"][target_index],
            )


if __name__ == "__main__":
    coordination()
    composition()
    transfer()
