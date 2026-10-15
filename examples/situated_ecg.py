"""Run and resume a discrete-time colour-board coordination demonstration."""

import json

from self_assembling_games.situated import create_backend, resolve_model, restore_backend


model = resolve_model("ecg.demo", {
    "realisation": {"provider": "scripted", "parameters": {
        "scripts": {"A": ["wait", "finish"], "B": ["right", "finish"]}}},
    "adaptation": {"provider": "reward_trace", "parameters": {}},
})
backend = create_backend(model, seed=12)
backend.step_stage()
snapshot = json.loads(json.dumps(backend.snapshot()))
resumed = restore_backend(model, snapshot)
backend.advance(3)
resumed.advance(3)
assert backend.snapshot() == resumed.snapshot()
print(json.dumps(resumed.inspect(), indent=2))
