# Multilocus realisations of games

Multilocus models connect an inherited allele combination to the organisation
that acts in a game. `population.multilocus` implements two established population
descriptions with the same selection and transmission operators:

| Method | Census state | Clock |
|---|---|---|
| `haplotype.deterministic` | Frequencies of ordered haplotypes | Discrete generations |
| `wright_fisher.haploid` | Integer counts of haploid genomes | Discrete generations |

The implementation uses exact finite expectations and deterministic population
recursions evaluated in floating-point arithmetic. Wright–Fisher sampling adds
finite-population drift. These methods use NumPy and require no ODE solver.

## Objects and interpretation

The four [model responsibilities](model.md) remain operative:

- **Game:** observations, legal choices, chance inputs and functional returns.
- **Organisation:** which encounter slots belong to the same sampled individual,
  and how their returns contribute to reproduction.
- **Realisation:** genome, genotype-to-plan expression, population state and
  transmission operations.
- **Specification:** initial choices, effective parameters and source declaration.

`Locus` declares an inherited coordinate and its possible alleles. `GenomeSpec`
retains locus and allele order and enumerates haplotypes lexicographically in that
order. A haplotype is the whole genotype of an individual in these haploid models.
`ExpressionMap` assigns a complete plan to every encounter slot for each haplotype.
`TransmissionSpec` reconstructs offspring allele combinations from selected
parents, then applies locus mutation.

Expression tables allow several loci to affect one component, one locus to affect
several components, and several genotypes to express the same functional plans.
The underlying genotype frequencies remain separate. A complete plan specifies
conditional acts over the finite game's declared decision contexts; it can be
used by several functional modules belonging to one individual. This first
expression model supplies deterministic plans and fixed inherited architectures.
Molecular expression, diploid dominance, architecture evolution and within-life
learning require their own mechanisms.

## Native construction

```python
from self_assembling_games import ModelSpecification, SourceDeclaration, build

specification = ModelSpecification.resolve(
    game="ML-COORD",
    game_parameters={},
    organisation={"individuals": {"cell": ["sender", "receiver"]}},
    realisation="population.multilocus",
    realisation_parameters={
        "method": "haplotype.deterministic",
        "genome": {"loci": [
            {"name": "s", "alleles": [0, 1]},
            {"name": "r", "alleles": [0, 1]},
        ]},
        "expression": {"plan_indices": {
            "sender": [1, 1, 2, 2],
            "receiver": [1, 2, 1, 2],
        }},
        "transmission": {
            "mutation": {
                "s": [[0.99, 0.01], [0.01, 0.99]],
                "r": [[0.99, 0.01], [0.01, 0.99]],
            },
            "recombination": {
                "kind": "independent_intervals", "fractions": [0.5],
            },
        },
        "fitness": {"background": 1, "scale": 1},
        "initialization": {"kind": "frequencies", "frequencies": [0.25] * 4},
    },
    source=SourceDeclaration(
        kind="deliberate_variant",
        identifiers=("constructed:intracellular-coordination",),
        assumptions=("Both functional modules act within one haploid cell.",),
    ),
)
simulation = build(specification, seed=7, protocol={"generations": 100})
first_generation = simulation.step()
simulation.run(generations=99)
print(simulation.model_instance.inspect())
```

The four haplotypes are `00, 01, 10, 11`. The sender and receiver each have four
complete binary plans, ordered `00, 01, 10, 11` over their two input contexts.
Plan indices 1 and 2 select identity and inversion, respectively. The expression
table therefore realises `message = state XOR s`, `act = message XOR r`.

Native and profile construction use the same gates. An `AdaptiveProfile` stores
`genome`, `expression`, `transmission`, `fitness` and optional `population_size`
in its `genetics` field; its `population` field holds the individual bindings.
Initialisation remains serializable in `protocol` for profile compatibility and
is resolved into the model's realisation specification. Outer `generations` and
integer `sample_interval` are supplied through `build(..., protocol=...)`.

To use Wright–Fisher, select `wright_fisher.haploid`, provide `population_size`,
and declare either:

```python
{"kind": "explicit_counts", "counts": [50, 50, 50, 50]}
# or:
{"kind": "multinomial_from_frequencies", "frequencies": [0.25] * 4}
```

`population_size=N` counts haploid census genomes. The second initialiser draws
an initial multinomial census; equal input frequencies can yield unequal counts.
Omitted initialisation uses equal frequencies for the deterministic method and
multinomial sampling from equal frequencies for Wright–Fisher. Initial counts,
frequencies and RNG state are retained. Deterministic models reject a census size.

## Participation, functional returns and reproductive weight

Individual bindings partition the game's encounter slots exactly once. Every
slot within a group expresses the same sampled genotype. Different groups draw
independently **with replacement** from the single population, so repeated census
identities and self encounters are possible. Reproductive parents also use
independent draws with replacement. These are distinct from the existing
complete-plan population layout and its matching contract.

Let `E_s(g)` be the plan expressed by genotype `g` at slot `s`, and let
`M_s(l,p)` indicate whether plan `p` is compatible with finite encounter leaf `l`.
For individual group `b`, define:

\[
C_b(g,l)=\prod_{s\in b}M_s(l,E_s(g)),\qquad
z_b(l)=\sum_g x_g C_b(g,l). \tag{1}
\]

A focal genotype fixes every slot of **one group at a time**. Conditional slot
returns integrate every other group through its `z`. For chance mass `q_l`,
leaf return `R_s(l)` and complete-plan cost `k_s`, this is:

\[
F_s(g,x)=\sum_l q_l C_b(g,l)
\prod_{d\ne b}z_d(l)R_s(l)-k_s(E_s(g)),\quad s\in b. \tag{2}
\]

The declared return weights `omega_s` are nonnegative and sum to one over all
slots; the default gives each slot equal weight. A zero-weight group may influence
other participants while contributing no direct reproductive return. Reproductive
weight is the explicit affine transformation:

\[
w_g(x)=a+\beta\sum_s\omega_sF_s(g,x). \tag{3}
\]

`fitness.background=a` and `fitness.scale=beta` are finite and nonnegative.
The resulting weights must be finite and nonnegative, with positive population
mean. Signed game returns and complete-plan costs can satisfy this condition
under a declared background. Validation applies no implicit payoff shift.

For games with two-role payoff matrices, those matrices supply returns in place
of leaf returns, following the existing finite-game contract. Slots in one group
index both plans from the same genotype; separately sampled groups integrate
against the other group's expressed-plan distribution. Equations (1)–(3) retain
intra-individual associations and define conditional fitness for absent genotypes.

## Generation and transmission

With a census immediately before the assay, each generation follows:

\[
y_g=\frac{x_gw_g(x)}{\sum_hx_hw_h(x)},\quad
v=R(y),\quad p=vQ. \tag{4}
\]

`U_i` is the row-stochastic mutation matrix for locus `i`, in declared allele
order. Rows identify parental alleles and columns offspring alleles.
Independent locus mutation gives `Q = U_1 tensor-product ... tensor-product U_L`.
Omitted matrices are identity matrices. The implementation contracts one locus
axis at a time, avoiding allocation of the full mutation kernel.

Three concrete recombination modes are available:

| Mode | Parameters | Parental reconstruction |
|---|---|---|
| `clonal` | None | Copy one selected haplotype |
| `independent_intervals` | Adjacent `fractions`, each in `[0, 0.5]` | Choose an initial parent equally, then independently switch parental source at each boundary |
| `single_boundary_exchange` | `exchange_probability=c` in `[0, 1]` | With probability `c`, choose one internal boundary uniformly and join one parent's prefix to the other's suffix |

Independent interval switches supply a declared recombination model for random
pairing, a transient diploid phase and meiosis. For two loci, with
`D_y = y00*y11 - y01*y10`, its update is:

\[
v=y+rD_y(-1,+1,+1,-1),\qquad D_v=(1-r)D_y. \tag{5}
\]

For more loci, the joint switch law is an explicit additional assumption.
Single-boundary exchange implements the supplied study's separate operator:

\[
R_c(y)=(1-c)y+\frac{c}{L-1}
\sum_{b=1}^{L-1}y_{1:b}\otimes y_{b+1:L}. \tag{6}
\]

It requires an internal boundary. `c` is an exchange-event probability; its full
allowed range describes this constructed exchange operation. Independent interval
switches and single-boundary exchange give different multi-locus offspring laws.
Parents sharing alleles can yield unchanged offspring after an exchange.

The deterministic next census is `x'=p`. Wright–Fisher instead draws:

\[
C'\sim\mathrm{Multinomial}(N,p),\quad x'=C'/N,\qquad
\mathbb E[x'\mid p]=p,\quad
\mathrm{Cov}(x'\mid p)=\frac{\mathrm{diag}(p)-pp^T}{N}. \tag{7}
\]

Counts are the authoritative finite state; frequencies are derived from them.
The proposed generation and next-census assay are validated before committing
state and RNG continuation. A failed `step()` retains the preceding census,
clock, cumulative measurements and last successful generation record.

## Inspection, measurements and continuation

`step()` returns the frequency vectors `before`, `after_selection`,
`after_recombination`, `expected_offspring` after mutation, and `after_census`,
plus the pre-reproduction assay. Inspection includes allele frequencies,
unstandardized biallelic pairwise linkage disequilibria, expression coordinates
and transmission. Complete haplotype frequencies also retain higher-order
associations that pairwise summaries omit.

`generation_average_expected_success` averages pre-reproduction assays over
**every advanced generation**, independently of measurement sampling.
`final_policy_success_probability` evaluates the present census. At generation
zero the cumulative average is undefined (`None`), while final evaluation is
available. `sample_interval` controls retained census assays. Inspection and
external evaluation preserve population and RNG state.

Checkpoints retain initial and current state, the last staged generation, RNG,
clock, phase history, measurements, source declaration, expression and game
hashes. `set_population` records a deliberate intervention, accepting frequencies
for deterministic models and counts for Wright–Fisher. Configurations also pin
locus/allele order and finite-game plan coordinates through the resolved game.
Custom factories must be registered again when restoring; `game=` overrides use
the same resolution checks.

```sh
sg list --family constructed
sg describe ml-coord.deterministic.demo --resolved --seed 7
sg run ml-coord.deterministic.demo --generations 100 --output runs/coord
sg run ml-transfer.wf.demo --seed 7 --generations 100 --output runs/transfer
sg evaluate runs/transfer
sg resume runs/transfer --generations 100
sg ensemble ml-transfer.wf.demo --seeds 1 2 3 --generations 100 \
  --workers 2 --output runs/transfer-ensemble
```

Resource bounds cover haplotypes, nonzero source patterns, estimated transmission
and generation contraction work, genotype compatibility masks, census
measurements and phase history. Defaults are 4096 haplotypes, 65536 source
patterns, 10 million transmission work units, 50 million generation work units,
10 million genotype mask entries, and 100000 measurements/history entries.
Only the last staged generation is retained in the checkpoint; applications can
consume each returned `step()` record for a full trajectory.

## Constructed assays and validation

Six demonstration profiles cover three games, each under both methods:

- `ML-COORD`: intracellular identity/inversion sender and receiver variants.
- `ML-TRANSFER`: a supplied, fixed NAND operation with two inherited adapters.
  The old route is retained by construction; template provenance identifies its
  truth table and the adapters as the changing components.
- `ML-COMPOSE`: preformed local interface variants; even parity coordinates the
  composite operation.

The [executable comparisons](../examples/multilocus_assays.py) expose shared
participation, staged transmission and target-dependent adapter accessibility.
The separable and coupled expression tables provide the same adapter repertoire.
Independent bit mutation in each encoding supplies different functional
neighbourhoods. A transported-coordinate null must transform expression,
initial state **and the entire mutation/recombination operation**. The independent
test oracle verifies that null; production transmission remains the three concrete
operators above. Category relabelling preserves Wright–Fisher transition laws,
while the same seed need not preserve category-by-category sample trajectories.

Independent parent-pair enumeration checks recombination, and explicit mutation
kernels check the axis contractions. Tests also cover shared-individual fitness,
matrix returns, neutral expression degeneracy, count conservation, multinomial
moments, native/profile equivalence, failure atomicity, clocks and continuation.
The existing 85-profile behavioural baselines remain intact.

For the manuscript's two-locus first generation, weights `1 + coordination`,
mutation `.01`, and a uniform founder give success `.6600666667`, `.6520633333`
and `.5800333333` at recombination fractions `0`, `.05` and `.5`. For the uniform
even-parity three-locus founder, single-boundary exchange `.5` gives success
`.75` after recombination and `.735298` after mutation `.01`. Stage names identify
the quantity compared.

The general engine uses ordinary floating-point recursions. An unstable symmetric
trajectory can amplify roundoff just as it amplifies a specified perturbation.
The study's symmetry-constrained long-run E2 value is a separate scalar reference;
the engine applies no symmetry projection. Explicit perturbations and analytical
invariants are useful controls when interpreting long-run comparisons.

## Sources and scope

[Kirkpatrick, Johnson and Barton (2002), *General models of multilocus evolution*](https://doi.org/10.1093/genetics/161.4.1727)
provides the general framework for multilocus selection, inheritance and mutation.
The publisher/institutional abstract establishes that scope; this implementation
uses the elementary finite-haplotype recursions specified above.

[Barton and Otto (2005), *Evolution of Recombination Due to Random Drift*](https://www.zoology.ubc.ca/~otto/Reprints/BartonOtto2005.pdf),
particularly the model description on p.2356 and Appendix A on pp.2367–2368,
specifies haploid selection followed by random mating/meiosis and multinomial
sampling of juveniles. Their notation samples `2N` haploid genomes; equation (7)
uses `N` for that haploid census count. Their focal study concerns recombination
modifiers; its specific fitness assumptions are separate from our game's
explicit transformation (3). Independent locus mutation is the declared extension
in (4).

The supplied working study, *Material realisations of self-assembling games*
(11 October 2026), supplies the constructed coordination, adapter and parity
assays and single-boundary operator. These demonstrate specified genetic
encodings. Molecular pathways, chemotaxis and ecological maintenance require
further realisation and environmental mechanisms. The [source register](sources.md)
keeps these constructed cases distinct from Barrett/Skyrms source reproductions.
