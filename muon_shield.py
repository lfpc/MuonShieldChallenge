import os
import sys
from pathlib import Path

import numpy as np
import torch

from utils import REPO_DIR, cavern_overlap, iron_cost, load_muons, to_geometry, total_length

CUDA_MUONS_DIR = Path(os.environ.get('CUDA_MUONS_DIR', REPO_DIR.parent / 'MuonsAndMatter' / 'cuda_muons'))
sys.path.insert(0, str(CUDA_MUONS_DIR))  # cuda_muons is imported in MuonShieldProblem.simulate

PARAM_NAMES = ('z_gap', 'dZ', 'dXIn', 'dXOut', 'dYIn', 'dYOut', 'gapIn', 'gapOut', 'dX_yokeIn', 'dX_yokeOut',
               'dY_yokeIn', 'dY_yokeOut', 'midGapIn', 'midGapOut', 'B')
DERIVED_PARAMS = ('dY_yokeIn', 'dY_yokeOut', 'midGapOut')
MUON_COLUMNS = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg_id')
OUTPUTS = ('n_hits', 'hits', 'muons')
DESIGNS = {
    'baseline': [
        [0.0, 120.5, 50.0, 50.0, 119.0, 119.0, 2.0, 2.0, 50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 1.9],
        [10.0, 250.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 72.0, 51.0, 72.0, 51.0, 0.0, 0.0, 1.9],
        [10.0, 250.0, 54.0, 38.0, 46.0, 130.0, 14.0, 9.0, 54.0, 38.0, 54.0, 38.0, 0.0, 0.0, 1.9],
        [10.0, 250.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 10.0, 31.0, 10.0, 31.0, 0.0, 0.0, 1.9],
        [10.0, 150.0, 5.0, 32.0, 54.0, 24.0, 8.0, 8.0, 5.0, 32.0, 5.0, 32.0, 0.0, 0.0, -1.9],
        [10.0, 150.0, 22.0, 32.0, 130.0, 35.0, 8.0, 13.0, 22.0, 32.0, 22.0, 32.0, 30.0, 30.0, -1.9],
        [10.0, 251.0, 33.0, 77.0, 85.0, 90.0, 9.0, 26.0, 33.0, 77.0, 33.0, 77.0, 0.0, 0.0, -1.9],
    ],
}

class MuonShieldProblem:
    def __init__(self, muons_file: str, sensitive_plane: dict, free_magnets: list[int],
                 free_params: list[str], bounds: dict[str, list[float]], design='baseline',
                 n_samples: int = 0, cavern: bool = False, n_steps: int = 5000, seed: int | None = None,
                 batch_size: int = 0, W0: float = 15e6, L0: float = 32.0, output: str = 'n_hits',
                 device: str = 'cuda'):
        """
        Args:
            muons_file: input muons (.npy or .h5).
            sensitive_plane: detector plane, a dict with 'position' (z), 'dx', 'dy', 'dz' (m). A muon hits the
                detector if it is inside its acceptance.
            free_magnets, free_params: optimised magnets and parameters (names in PARAM_NAMES).
            bounds: (low, high) of each free parameter, the same for every free magnet.
            design: reference design, a name in DESIGNS or a list of 15-parameter rows.
            n_samples: number of muons used (the first n_samples of the file); 0 uses all.
            cavern: simulate the cavern walls and require the magnets to fit inside them.
            n_steps: maximum number of propagation steps.
            seed: propagation seed. None draws a new seed at every evaluation (noisy objective).
            batch_size: muons propagated at once; 0 propagates all together.
            W0: maximum cost of the shield.
            L0: maximum length of the shield (m).
            output: what the objective returns, one of OUTPUTS (see `objective`).
            device: GPU used for the simulation.
        """
        if output not in OUTPUTS:
            raise ValueError(f'output must be one of {OUTPUTS}, got {output!r}')
        self.design = torch.tensor(DESIGNS[design] if isinstance(design, str) else design, dtype=torch.float64)
        invalid = [p for p in free_params if p not in PARAM_NAMES or p in DERIVED_PARAMS]
        if invalid:
            raise ValueError(f'Not optimisable parameters: {invalid}')
        self.free_index = torch.tensor([[m, PARAM_NAMES.index(p)] for m in free_magnets for p in free_params])
        self.dim = len(self.free_index)
        self.initial_phi = self.design[tuple(self.free_index.T)]

        self.bounds = torch.tensor([bounds[p] for m in free_magnets for p in free_params], dtype=torch.float64).T
        outside = (self.initial_phi < self.bounds[0]) | (self.initial_phi > self.bounds[1])
        if outside.any():
            raise ValueError(f'Reference design outside the bounds at (magnet, param): {self.free_index[outside].tolist()}')

        self.muons_file, self.n_samples, self.sensitive_plane = muons_file, n_samples, sensitive_plane
        self.cavern, self.n_steps, self.seed, self.batch_size = cavern, n_steps, seed, batch_size
        self.W0, self.L0, self.output, self.device = W0, L0, output, device
        self._muons = None

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> 'MuonShieldProblem':
        return cls(**config, **kwargs)

    @property
    def muons(self) -> torch.Tensor:
        if self._muons is None:
            self._muons = load_muons(self.muons_file, self.n_samples)
            self.n_samples = len(self._muons)  # all the muons of the file if n_samples was 0 (or more than the file has)
        return self._muons

    def design_from_phi(self, phi) -> torch.Tensor:
        """Full design (n_magnets, 15) from the free parameters (dim,) or from a flattened full design."""
        phi = torch.as_tensor(phi, dtype=torch.float64).detach().cpu().flatten()
        if phi.numel() == self.design.numel():
            return phi.view_as(self.design).clone()
        design = self.design.clone()
        design[tuple(self.free_index.T)] = phi
        return design

    def constraints(self, phi) -> torch.Tensor:
        """Constraint residuals g(phi), feasible if all <= 0. Cheap, no simulation.

        - length - L0 (m)
        - (cost - W0)
        - if cavern: overlap of each magnet with the cavern walls (m), 4 per magnet
        """
        design = self.design_from_phi(phi)
        geometry = to_geometry(design)
        g = torch.tensor([total_length(design) - self.L0, (iron_cost(geometry) - self.W0)], dtype=torch.float64)
        return torch.cat([g, cavern_overlap(geometry).flatten()]) if self.cavern else g

    def simulate(self, phi) -> torch.Tensor:
        """Every input muon after the simulation, in the input order, (N, 7): px, py, pz, x, y, z, pdg_id."""
        from cuda_muons import run_from_params  # needs a GPU and the faster_muons_torch extension
        if torch.device(self.device).index is not None:  # the kernels run on the current device
            torch.cuda.set_device(self.device)
        geometry = to_geometry(self.design_from_phi(phi)).numpy()
        seed = self.seed if self.seed is not None else np.random.randint(2**30)
        batches = self.muons.split(self.batch_size) if self.batch_size > 0 else [self.muons]
        out = [run_from_params(geometry, batch, sensitive_plane=self.sensitive_plane, n_steps=self.n_steps,
                               add_cavern=self.cavern, return_all=True, seed=seed + i, device=self.device,
                               histogram_dir=str(CUDA_MUONS_DIR / 'data'))
               for i, batch in enumerate(batches)]
        return torch.cat([torch.stack([o[k].float() for k in MUON_COLUMNS], 1) for o in out])

    def hits(self, muons: torch.Tensor) -> torch.Tensor:
        """(N,) True for each simulated muon inside the acceptance of the sensitive plane."""
        plane = self.sensitive_plane
        return ((muons[:, 3].abs() < plane['dx'] / 2) & (muons[:, 4].abs() < plane['dy'] / 2)
                & (muons[:, 5] >= plane['position'] - plane['dz'] / 2))

    def objective(self, phi) -> float | torch.Tensor:
        """Objective f(phi), depending on `output`:

        - 'n_hits': number of muons hitting the detector
        - 'hits': (N,) 1 for each input muon hitting the detector, 0 otherwise
        - 'muons': the muons hitting the detector after the simulation, see `simulate`
        """
        muons = self.simulate(phi)
        hits = self.hits(muons)
        if self.output == 'hits':
            return hits.float()
        if self.output == 'muons':
            return muons[hits]
        return float(hits.sum())

    def evaluate(self, phi) -> dict:
        """Objective, constraint residuals and additional information on one design."""
        design = self.design_from_phi(phi)
        hits = self.simulate(design)
        return hits

    def __call__(self, phi) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        """(f, g) of phi (dim,), or of a batch (n, dim): f stacked (a list for output 'muons'), g (n, n_constraints)."""
        phi = torch.as_tensor(phi, dtype=torch.float64)
        if phi.dim() > 1:
            f, g = zip(*(self(p) for p in phi))
            return list(f) if self.output == 'muons' else torch.stack(f), torch.stack(g)
        return torch.as_tensor(self.objective(phi)), self.constraints(phi)


if __name__ == '__main__':  # test: simulate the reference design on the first n_samples muons of a config
    import argparse

    from utils import load_config

    parser = argparse.ArgumentParser(description='Simulate the reference design and print the number of hits.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=str(REPO_DIR / 'configs' / 'easy.json'), help='problem configuration')
    parser.add_argument('--n_samples', type=int, default=100_000, help='number of muons; 0 uses all the muons of the file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    args = parser.parse_args()

    problem = MuonShieldProblem.from_config(load_config(args.config, n_samples=args.n_samples, output='n_hits'),
                                            device=f'cuda:{args.gpu}')
    n_hits, g = problem(problem.initial_phi)
    print(f'n_hits = {n_hits.item():.6g}, constraints = {g.tolist()}')
