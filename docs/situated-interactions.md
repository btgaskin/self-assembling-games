# Situated interactions

A situated model samples a world and lets identified actors act through successive
stages. The interaction supplies each actor's observation and legal actions. A
local controller uses that observation and its retained memory to choose an
action. A separate adaptation component receives local feedback and can update
both policy memory and its own adaptive state. Actor identities carry ownership;
communicative roles and message channels can be added by a particular interaction.

`ecg.demo` provides a runnable colour-board coordination fixture inspired by
Scott-Phillips, Kirby and Ritchie (2009), *Signalling signalhood and the emergence
of communication*, [doi:10.1016/j.cognition.2009.08.009](https://doi.org/10.1016/j.cognition.2009.08.009).
The fixture is a **constructed discrete-time variant**. It supplies random and
explicit scripted controllers to qualify the mechanics. These controllers do not
model human recognition of signalhood or explain the acquisition of communication.

## The colour-board fixture

Each encounter draws two independent boards with iid uniform colour cells,
conditioned jointly on the boards sharing at least one colour. The default is two
2×2 boards drawn from red, green, blue and yellow. Each actor starts at a uniformly
sampled cell. Its observation contains its own board, both actors' positions and
finished flags, the board size, and the current tick. The other board remains
private until both actors finish.

At each tick, every unfinished actor chooses `up`, `down`, `left`, `right`, `wait`
or `finish`. All choices use observations of the same pre-tick world. A boundary
move keeps the actor in place. `finish` locks its current cell and removes it from
subsequent action batches. Once both finish, both boards and chosen colours are
revealed in feedback. Both actors receive 1 when their chosen colours match and
0 otherwise. The default tick cap is 64. Reaching the cap while an actor remains
unfinished truncates the encounter, with no terminal reward or board reveal.
Finishing together on the final permitted tick terminates normally.

Movement can affect another actor's future observations. The interaction defines
all six choices as ordinary world actions and assigns none a special message
status. The synchronous timing, finish action, stopping cap, board law and supplied
policies are explicit model choices recorded in the model's source declaration.

```python
from self_assembling_games.situated import create_backend, resolve_model, restore_backend

model = resolve_model("ecg.demo", {
    "realisation": {
        "provider": "scripted",
        "parameters": {"scripts": {"A": ["wait", "finish"],
                                    "B": ["right", "finish"]}},
    },
})
backend = create_backend(model, seed=12)
backend.step_stage()                 # one simultaneous tick; encounter still open
saved = backend.snapshot()           # JSON-serializable, including the current world
continued = restore_backend(model, saved)
continued.advance(3)                 # complete three attempted encounters
print(continued.evaluate())
```

The execution clock `encounters` counts completed attempts, including truncations.
`position` increases only when an encounter ends. `advance(n)` completes `n`
encounters from the current position, finishing an already-open encounter first.
`step_stage()` exposes individual batches for inspection and mid-encounter saves.
After an encounter ends, its terminal world stays available until the next stage
starts a freshly sampled world. Policy memory, adaptive state, observed histories
and the backend's history persist across this reset. A scripted controller consumes
its script across encounters and defaults to `wait` after its script ends.

## Custom interactions and local policies

`Interaction` defines five methods plus an actor identity sequence:

```python
class MyInteraction:
    actors = ("one", "two")

    def initial_world(self, rng): ...             # JSON-serializable dictionary
    def stage(self, world): ...                   # Stage(id, actors)
    def observations(self, world): ...            # actor -> private observation
    def legal_actions(self, world, actor): ...    # indexable action space
    def transition(self, world, actions, rng): ... # StageResult
```

`Stage` lists the actors that choose simultaneously at that stage. Sequential
interactions use one actor in a stage; different stages may activate different
subsets. The runtime builds all views before calling controllers and commits the
complete action batch in one `transition`. `StageResult` carries the next world,
actor-keyed feedback, separate `terminated` and `truncated` flags, and public
inspection information. Interaction methods receive detached copies of the world.
All retained mutable interaction state belongs in `world`; the interaction object
holds its fixed configuration.

Each controller implements
`choose(view, legal_actions, state, rng) -> (action, new_state)`. Its view contains
its actor identity, encounter index, stage name, private observation and its own
observable history. Each history entry contains its own observation, own action
when active, own feedback and encounter end flags. Controllers receive copies of
these values and their retained state. The hidden world, other controllers'
memory, their private feedback and the backend's inspection history are outside
this interface. A controller may mutate its input copies freely.

The separate adaptation component implements
`update(event, controller_state, adaptation_state, rng) -> (controller_state, adaptation_state)`.
It receives the same actor-local event recorded in observed history, including
feedback received while inactive. `none` preserves both states. `reward_trace`
retains a feedback count and reward sum as an example of independent adaptive
state; it preserves the controller's decisions. Custom adapters can implement a
specified learning rule through explicit state updates.

Action spaces support `len`, indexed sampling and membership testing. Lazy ranges
and custom indexable spaces allow large sampled interactions without constructing
a full joint policy representation. Observations, chosen actions, state and
feedback must round-trip through JSON. Colour names and symbolic action labels
retain their given spelling and meaning.

Importable factories make components reconstructible in a fresh process:

```python
model = resolve_model("ecg.demo", {
    "interaction": {"provider": "my_package.world:make_interaction",
                    "game": "my_world", "parameters": {}},
    "realisation": {"provider": "my_package.policy:make_controller",
                    "parameters": {}},
    "adaptation": {"provider": "my_package.learning:make_adapter",
                   "parameters": {}},
})
```

Each factory accepts its parameter dictionary and returns the relevant object.
Every controller and adapter instance belongs to one actor. Keep mutable memory
in the explicit returned state, rather than in the factory object, so restoration
captures it. The same importable modules must be available when restoring.

Snapshots retain model identity and parameters, current world and stage position,
completed encounter count, full history, each actor's observed history, separate
controller and adaptive states, and random generator states. Independent random
streams serve world sampling, each controller and each adapter. Restoring checks
model equality and recovers those streams. Evaluation reads retained outcomes and
does not draw actions or modify the world. The common experiment runner can use
this backend through its `advance`, `snapshot`, `evaluate` and `inspect` methods.

## Catalogue coverage

The Barrett catalogue retains its existing case and source-question identifiers.
The `ECG` situated entry and `ecg.demo` preset add this source-linked fixture as a
constructed variant. The library continues to list KIND as analysis, and POLY,
SUB, DISC, GEN, SET and RIT as cases awaiting a concrete operational protocol.
Their presence records the book's scope while their readiness fields distinguish
implemented mechanisms from proposed extensions. The ECG fixture adds a general
world/action/feedback substrate; it does not resolve those source questions.
