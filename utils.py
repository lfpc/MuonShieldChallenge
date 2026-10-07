"""Configuration and muon loading, shield geometry, length, iron cost and cavern overlap."""
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent

IRON_COST = 7870.0 * (2.0 + 6.0)  # aisi1010, per m^3: density (kg/m^3) * (material + manufacturing cost per kg)
CAVERN_Z, CAVERN_X, CAVERN_Y = 1837.8, (356.0, 456.0), (169.0, 335.0)


def load_config(path: str, **overrides) -> dict:
    """Read a JSON configuration, whose keys are the arguments of MuonShieldProblem.

    Overrides that are not None replace the values of the file. muons_file may contain $VARS and is
    relative to the repository.
    """
    with open(path) as f:
        config = json.load(f)
    config.pop('description', None)
    config.update({k: v for k, v in overrides.items() if v is not None})
    config['muons_file'] = str((REPO_DIR / Path(os.path.expandvars(config['muons_file'])).expanduser()).resolve())
    return config


def sample_muons(spec: str, n_samples: int, chunk: int = 10_000_000) -> np.ndarray:
    """n_samples muons from MuonsGenerator, as described by the JSON file spec: {'gmm': GMM file, 'z': Beta fit of z
    (paths relative to spec), 'seed'}. Sampled in chunks, chunk i with seed seed + i, so fewer muons are the first ones
    of more."""
    from MuonsGenerator import MuonsGenerator
    if n_samples <= 0:
        raise ValueError(f'n_samples must be set to sample muons from {spec}')
    with open(spec) as f:
        s = json.load(f)
    base = Path(spec).parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = MuonsGenerator.from_file(str(base / s['gmm']), str(base / s['z']), device=device)
    return np.concatenate([generator.sample(chunk, seed=s['seed'] + i // chunk).cpu().numpy()  # whole chunks, as
                           for i in range(0, n_samples, chunk)])[:n_samples]  # a draw depends on its size


def load_muons(path: str, n_samples: int = 0) -> torch.Tensor:
    """First n_samples muons (0: all) of a .npy or .h5 file, or n_samples muons sampled from MuonsGenerator if path
    is a JSON description of it (see sample_muons): px, py, pz, x, y, z, pdg_id. Other columns are ignored."""
    t0 = time.time()
    if path.endswith('.json'):
        muons = sample_muons(path, n_samples)
    elif path.endswith('.npy'):
        muons = np.array(np.load(path, mmap_mode='r')[:n_samples or None, :7], dtype=np.float32)
    else:
        with h5py.File(path, 'r') as f:
            n = len(f['px']) if n_samples <= 0 else min(n_samples, len(f['px']))
            keys = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg')
            muons = np.empty((n, len(keys)), np.float32)
            for j, key in enumerate(keys):
                muons[:, j] = f[key][:n]
    print(f'{"Sampled" if path.endswith(".json") else "Loaded"} {len(muons):,} muons from {path} in {time.time() - t0:.1f} s')
    return torch.from_numpy(muons)


def to_geometry(design: torch.Tensor) -> torch.Tensor:
    """Design in the cuda_muons layout: derived entries filled."""
    geometry = design.clone()
    geometry[:, 10:12] = design[:, 8:10]
    geometry[:, 13] = geometry[:, 12]
    return geometry


def total_length(design: torch.Tensor) -> float:
    """Shield length (m)."""
    return (design[:, 0].sum() + 2 * design[:, 1].sum()).item() / 100


def iron_cost(geometry: torch.Tensor) -> float:
    """Cost of the iron of the magnets, with the blocks built by cuda_muons (its 0.1 mm anti-overlap ignored).

    The section of a magnet (cm^2) is 2 dX (2 dY + dY_yoke) for the core, 2 dY_yoke (dX + x_yoke + 2 gap) for
    the top yokes and 2 x_yoke (2 dY + dY_yoke) for the return yokes. It is quadratic in z, so Simpson's rule
    gives the exact volume.
    """
    def area(dx, dy, gap, x_yoke, dy_yoke):
        return 2 * (dx * (2 * dy + dy_yoke) + dy_yoke * (dx + x_yoke + 2 * gap) + x_yoke * (2 * dy + dy_yoke))

    face_in, face_out = geometry[:, [2, 4, 6, 8, 10]].T, geometry[:, [3, 5, 7, 9, 11]].T
    volume = 2 * geometry[:, 1] * (area(*face_in) + 4 * area(*(face_in + face_out) / 2) + area(*face_out)) / 6
    return volume.sum().item() * 1e-6 * IRON_COST


def cavern_overlap(geometry: torch.Tensor) -> torch.Tensor:
    """Overlap (m) of each magnet with the cavern walls, (n_magnets, 4): x_in, x_out, y_in, y_out."""
    z_out = torch.cumsum(geometry[:, 0] + 2 * geometry[:, 1], 0)
    upstream = torch.stack([z_out - 2 * geometry[:, 1], z_out], 1) <= CAVERN_Z
    x = geometry[:, 2:4] + geometry[:, 6:8] + geometry[:, 8:10] + geometry[:, 12:14]
    y = geometry[:, 4:6] + geometry[:, 10:12]
    return torch.cat([x - torch.where(upstream, *CAVERN_X), y - torch.where(upstream, *CAVERN_Y)], 1) / 100
