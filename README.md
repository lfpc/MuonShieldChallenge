# Muon Shield Challenge

A self-contained optimisation problem for the SHiP muon shield. A design (magnet
dimensions and fields) goes in. Out come an objective, which counts the muons
that reach the detector, and constraint residuals (length, cost, cavern).

Muons are propagated on the GPU with `run_from_params` of the `cuda_muons` framework from
[MuonsAndMatter](../MuonsAndMatter/cuda_muons). Only that framework is used:
there is no Geant4 and no FEM field simulation. Each iron block carries a
uniform field.

| File | Content |
|---|---|
| `muon_shield.py` | `MuonShieldProblem`: design, bounds, constraints and simulation. Run it to test the setup. |
| `run.py` | evaluates one design (the reference or one from a file): hits, length, cost, constraints |
| `utils.py` | config and muon loading, geometry, length, iron cost, cavern overlap |
| `MuonsGenerator.py` | muon sampler from a Gaussian mixture on (pz, pt) |
| `configs/` | the problem definitions (`easy.json`, `hard.json`) |
| `baselines/` | black-box optimisers (random search, GA, DE, CMA-ES, BO, TuRBO) and their comparison |

## Setup

You need an NVIDIA GPU and `MuonsAndMatter` next to this folder (otherwise set
`CUDA_MUONS_DIR=/path/to/MuonsAndMatter/cuda_muons`). The muon files are read from
`MuonsAndMatter/data/muons`.

**1. Environment.** `environment.yml` has Python, torch 2.7.1 (CUDA 12.6) and a
matching CUDA 12.6 toolchain with gcc 13 to build the extension:

```bash
conda env create -f environment.yml                  # or: conda env create -f environment.yml -p /path/to/envs/muonshield
conda activate muonshield
```

**2. Build the CUDA extension** `faster_muons_torch` (once per environment):

```bash
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation ../MuonsAndMatter/cuda_muons/faster_muons_torch
```

Check it from a folder other than `cuda_muons/`, where the source folder of the same
name would be imported instead. It should print the path of a `.so` file:

```bash
cd ~ && python -c "import torch, faster_muons_torch as f; print(torch.cuda.get_device_name(), f.__file__)"
```

**3. Run a simulation** of the reference design:

```bash
python muon_shield.py                                            # easy config, first 100k muons
python muon_shield.py --n_samples 0                              # all the muons of the file
python muon_shield.py --config configs/hard.json --n_samples 1000000 --gpu 1
```

It prints the number of hits and the constraint residuals.

### Troubleshooting the build

| Error | Cause and fix |
|---|---|
| `unsupported GNU version! gcc versions later than 13 are not supported` | gcc too new for nvcc 12.x: use the `gxx=13` of `environment.yml`. |
| `nv/target: No such file or directory` | CUDA headers not on the include path: `export NVCC_APPEND_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include"` and build again. |
| `curand_kernel.h: No such file or directory` | install `libcurand-dev` (same `cuda-version` as the rest). |
| `No module named 'torch'` during the build | torch is not installed in the active environment. |
| CUDA version mismatch with torch | `nvcc --version` must have the same major version as `torch.version.cuda`; mixed CUDA packages from different channels in one environment are the usual cause. |

Any environment works as long as it has a CUDA build of torch, numpy, h5py, scipy and pandas
(`requirements.txt`) and an `nvcc` matching torch's CUDA version with a gcc it supports.

## Configurations

| | `configs/easy.json` | `configs/hard.json` |
|---|---|---|
| free parameters | 30: `dZ, dXIn, dXOut, gapIn, gapOut` of magnets 1-6 | 72: all independent parameters of magnets 1-6 |
| cavern | no | yes: walls are simulated, and magnets must fit inside |
| noise | fixed seed (deterministic) | new seed per evaluation |
| muons | first 10⁷ of `full_sample_after_target.h5` | first 10⁸ of `full_sample_after_target.h5` |
| constraints | length `<= L0`, cost `<= W0` | length, cost, cavern walls |

Every key of a config is an argument of `MuonShieldProblem`, documented in the docstring of
`MuonShieldProblem.__init__`. `muons_file` may contain `$VARS` and is relative to this
folder. To define a new problem, copy a config and edit it. `load_config` takes overrides:
`load_config('configs/easy.json', n_samples=100_000, seed=3)`.

## Parametrisation

The reference design (`baseline`) has 7 magnets, each with 15 parameters. Lengths are
in cm and B is in T.

```
z_gap dZ dXIn dXOut dYIn dYOut gapIn gapOut dX_yokeIn dX_yokeOut dY_yokeIn dY_yokeOut midGapIn midGapOut B
```

- `z_gap` is the gap to the previous magnet, `dZ` the half-length, `dX`/`dY` the core
  half-width/height at the entrance (`In`) and exit (`Out`), `gap` the gap between core
  and return yoke (coils), `midGap` the gap between the cores at x = 0.
- `dX_yoke` is the return-yoke width. The top-yoke height `dY_yoke` equals
  that width, and `midGapOut = midGapIn`. These derived entries are never optimised.
- The sign of `B` gives the magnet's polarity, which can flip during the optimisation.
- Magnet 0, the hadron absorber, is fixed.

## Objective and constraints

```
minimise   f(phi) = number of muons hitting the detector
subject to g(phi) <= 0, with
           length - L0                    (m)
           cost - W0
           overlap with the cavern walls  (m, 4 per magnet: x and y at entrance and exit, only if cavern)
```

A muon hits the detector if it is inside the acceptance (`dx` x `dy`) of the
sensitive plane. `f` and `g` are returned raw: no penalty is combined with the
objective, and infeasible designs are simulated like any other. How to handle
the constraints is up to the optimiser. `g` is cheap to compute (no simulation).

The config key `output` selects what `f` is:

| `output` | `f(phi)` |
|---|---|
| `n_hits` (default) | number of muons hitting the detector |
| `hits` | `(N,)` vector, 1 for each input muon hitting the detector, 0 otherwise |
| `muons` | the muons hitting the detector after the simulation: px, py, pz, x, y, z, pdg_id |

**Cost.** The cost is the cost of the iron alone (aisi1010, 8 per kg), computed in
closed form from the magnet parameters with the same blocks that are simulated
(ARB8s, exact volume). The electrical cost used in BlackBoxOptimization needs `snoopy`,
so it is left out, and `W0` is scaled to match: the baseline costs about 7.4e6.

## Muons

A muon file (`.npy` array or `.h5` with one dataset per column) has the columns
`px, py, pz` (GeV), `x, y, z` (m) and `pdg` (±13). Other columns are ignored.
`n_samples` takes the first muons of the file; `0`, or more than the file has, uses
all of them, and `problem.n_samples` is set to the number loaded.

`MuonsGenerator` samples muons instead: (pz, pt) from a Gaussian mixture fitted on a
standardised space (log(pz, pt), or (logit of |p| in [p_min, p_max], log(pt/pz)) for the models of
`generator_muons/gmm/train_gmm_momentum.py`), px and py from a uniform azimuth, the production point on a
Gaussian ring, z from a Beta distribution, and the charge ±1 with probability 0.5.

```python
from MuonsGenerator import MuonsGenerator

gen = MuonsGenerator(weights, means, covariances, z_beta, mean, std, device='cuda')   # GMM (K,), (K,2), (K,2,2)
gen = MuonsGenerator.from_file('configs/gmm_model.json', 'configs/z_beta_fit.json')  # z_beta: a, b, loc, scale (cm)
muons = gen.sample(5_000_000, seed=1)              # (N, 7): px, py, pz, x, y, z, pdg_id
```

`python MuonsGenerator.py --n_samples 5000000 --output muons.npy` samples from
`configs/gmm_model.json` (or a model file given as first argument) and z from
`configs/z_beta_fit.json` (`--z_model`), and writes a file to use as
`muons_file`. It also plots the z, y vs x and pt vs pz histograms to `muons.png` (`--plot ''`
to skip, needs matplotlib).

## Python API

```python
from muon_shield import MuonShieldProblem
from utils import load_config

problem = MuonShieldProblem.from_config(load_config('configs/easy.json'), device='cuda:0')
problem.dim, problem.bounds, problem.initial_phi   # (dim,), (2, dim), (dim,)
f, g = problem(phi)                                # phi: (dim,), or a batch (n, dim) -> f stacked, g (n, n_constraints)
f = problem.objective(phi)                         # simulation, see `output`
g = problem.constraints(phi)                       # feasible iff all g <= 0 (cheap, no simulation)
out = problem.simulate(phi)                        # every muon after the simulation, input order
mask = problem.hits(out)                           # (N,) True for the muons hitting the detector
design = problem.design_from_phi(phi)              # full (7, 15) design
```

`phi` can also be a full flattened design (105 values) instead of the free parameters.

## Baselines

`baselines/` has common black-box optimisers, to run on a config and compare:

| method | |
|---|---|
| `random_search` | Gaussian perturbations of the reference design |
| `ga` | genetic algorithm: tournament selection, blend crossover, Gaussian mutation, elitism |
| `de` | differential evolution, DE/rand/1/bin |
| `cmaes` | CMA-ES (`cma`) |
| `bo` | Bayesian optimisation: GP + LogEI (`botorch`) |
| `turbo` | BO in an adaptive trust region (TuRBO-1) |
| `lcso_gd` | LCSO: per-muon hit classifier trained on many undersampled designs, trust-region step on its linear model with the exact constraints (SLSQP) (`baselines/lcso.py`) |
| `lcso_newton` | LCSO with the quadratic model (trust-region Newton step) |

```bash
pip install cma botorch                                                  # only for cmaes, bo and turbo
python baselines/run.py cmaes configs/easy.json --budget 200 --seed 0    # -> run cmaes_seed0 in results/baselines/easy.json
python baselines/run.py ga configs/easy.json --options '{"pop_size": 30}'
python baselines/compare.py results/baselines/easy.json                  # best f vs muons simulated -> easy.png
python baselines/run_all.py configs/easy.json --gpus 0 1 2 3 --n_repeats 3   # every method n_repeats times, then compare
```

The optimisers work in the unit cube of the bounds and start from the reference design, which is always the
first simulation: almost no design drawn uniformly in the bounds is feasible (0.2% for `easy`, none for `hard`).
The budget counts simulated muons, in full simulations: `--budget 200` is 200 × `n_samples` muons. LCSO spends most
of it on simulations of 1/16 of the muons (`resample_factor`), so it makes many more simulations, and the
comparison is plotted against the number of muons simulated. Only simulations on all the muons give a best f. The
constraints are cheap, so infeasible designs are not simulated and cost nothing: they get `f = n_samples * (1 +
violation)`, worse than any feasible design. BO, TuRBO and LCSO only propose feasible designs. With a noisy config
(`seed: null`), the best f of a run is optimistic: re-simulate the best design to compare.