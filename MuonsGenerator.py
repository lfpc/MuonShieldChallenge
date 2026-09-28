import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

class MuonsGenerator:
    """Samples muons (N, 7): px, py, pz (GeV), x, y, z (m), pdg_id.

    - (pz, pt) from a Gaussian mixture fitted on a standardised space x = (f(pz, pt) - mean) / std, where f is
      (log pz, log pt) for space 'log', or (logit((|p| - p_min) / (p_max - p_min)), log(pt / pz)) for space 'logit'
    - (px, py) from pt and a uniform azimuth
    - (x, y) on a ring of radius `radius` smeared by a Gaussian of width `sigma`
    - z = (loc + scale * Beta(a, b)) / 100, the Beta being fitted in cm
    - charge +1 or -1 with probability 0.5 (pdg_id = -13 * charge)
    Muons outside pz >= 0, pt >= 0, |p| <= p_max (the range of the propagation tables) are redrawn.
    """

    def __init__(self, weights, means, covariances, z_beta: dict, mean=(0.0, 0.0), std=(1.0, 1.0),
                 space: str = 'log', radius: float = 0.05, sigma: float = 0.016, p_min: float = 0.0,
                 p_max: float = 400.0, device: str = 'cpu'):
        """
        Args:
            weights, means, covariances: GMM parameters, shapes (K,), (K, 2), (K, 2, 2), columns (pz, pt).
            z_beta: Beta distribution of the production z (cm), dict with 'a', 'b', 'loc', 'scale'.
            mean, std: standardisation of the fit space the GMM was fitted with.
            space: fit space of the GMM, 'log' or 'logit' (see the class docstring).
            radius, sigma: ring radius and Gaussian smearing of the production point (m).
            p_min, p_max: momentum range of the 'logit' space (GeV). p_max is also the maximum muon momentum.
            device: device the muons are sampled on.
        """
        tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.cdf = tensor(weights).cumsum(0) / tensor(weights).sum()
        self.means = tensor(means)
        self.scale_tril = torch.linalg.cholesky(tensor(covariances))
        self.mean, self.std = tensor(mean).flatten(), tensor(std).flatten()
        self.z_beta, self.space, self.radius, self.sigma = z_beta, space, radius, sigma
        self.p_min, self.p_max = p_min, p_max
        self.device = device

    @classmethod
    def from_file(cls, path: str, z_path: str, **kwargs) -> 'MuonsGenerator':
        """From a JSON file with the GMM parameters 'weights', 'means', 'covariances', the standardisation
        'mean' and 'std', and for the 'logit' space 'columns' = ['logit_p', 'log_tan_theta'], 'p_min' and 'p_max'
        (as written by generator_muons/gmm), other keys ignored, and a JSON file with the z Beta parameters
        'a', 'b', 'loc', 'scale'."""
        with open(path) as f:
            model = json.load(f)
        with open(z_path) as f:
            z_beta = json.load(f)
        if model.get('columns') == ['logit_p', 'log_tan_theta']:
            kwargs = {'space': 'logit', 'p_min': model['p_min'], 'p_max': model['p_max'], **kwargs}
        return cls(model['weights'], model['means'], model['covariances'], z_beta, model['mean'], model['std'],
                   **kwargs)

    def sample_pz_pt(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """n (pz, pt) pairs from the GMM, not restricted to the physical range."""
        rand = lambda *shape: torch.rand(*shape, generator=generator, device=self.device)
        k = torch.searchsorted(self.cdf, rand(n)).clamp(max=len(self.cdf) - 1)
        e0, e1 = torch.randn(2, n, generator=generator, device=self.device)
        l00, l10, l11 = self.scale_tril[:, [0, 1, 1], [0, 0, 1]][k].T
        pz = self.means[k, 0] + l00 * e0
        pt = self.means[k, 1] + l10 * e0 + l11 * e1
        u, v = self.mean[0] + self.std[0] * pz, self.mean[1] + self.std[1] * pt
        if self.space == 'log':
            return u.exp(), v.exp()
        p = self.p_min + (self.p_max - self.p_min) * u.sigmoid()
        pz = p / (1 + (2 * v).exp()).sqrt()
        return pz, pz * v.exp()

    def sample(self, n: int, seed: int | None = None) -> torch.Tensor:
        generator = torch.Generator(self.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        rand = lambda: torch.rand(n, generator=generator, device=self.device)
        randn = lambda: torch.randn(n, generator=generator, device=self.device)

        pz, pt = self.sample_pz_pt(n, generator)
        for _ in range(100):
            out = (pz < 0) | (pt < 0) | (pz ** 2 + pt ** 2 > self.p_max ** 2)
            if not out.any():
                break
            pz[out], pt[out] = self.sample_pz_pt(int(out.sum()), generator)
        else:
            raise RuntimeError(f'{int(out.sum())} muons still outside pz >= 0, pt >= 0, |p| <= {self.p_max}')

        phi_p, phi_r = 2 * math.pi * rand(), 2 * math.pi * rand()
        x = self.radius * phi_r.cos() + self.sigma * randn()
        y = self.radius * phi_r.sin() + self.sigma * randn()
        # torch's Beta takes no generator: numpy, seeded from it
        z_rng = np.random.default_rng(int(torch.randint(2**62, (), generator=generator, device=self.device)))
        z = self.z_beta['loc'] + self.z_beta['scale'] * torch.from_numpy(z_rng.beta(self.z_beta['a'], self.z_beta['b'], n))
        z = z / 100  # cm -> m
        charge = 2 * torch.bernoulli(torch.full((n,), 0.5, device=self.device), generator=generator) - 1
        return torch.stack([pt * phi_p.cos(), pt * phi_p.sin(), pz, x, y, z.float().to(self.device), -13 * charge], 1)

    __call__ = sample


def plot_muons(muons: np.ndarray, path: str):
    """z (1D) and y vs x in cm, pt vs pz (2D, log colour scale) histograms of muons (N, 7), saved to path."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fontsize = 14
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    axes[0].hist(100 * muons[:, 5], bins=100, histtype='step', color='C0')
    axes[0].set_xlabel('z [cm]', fontsize=fontsize)
    axes[0].set_ylabel('muons', fontsize=fontsize)
    axes[0].set_title('Production z', fontsize=fontsize)

    xy = 100 * muons[:, 3:5]  # m -> cm
    lim = float(np.percentile(np.abs(xy), 99.9))
    xy_bins = np.linspace(-lim, lim, 201)
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xy_bins, xy_bins], density=True)
    im = axes[1].pcolormesh(xy_bins, xy_bins, H.T, cmap='viridis', norm=LogNorm())
    axes[1].set_xlabel('x [cm]', fontsize=fontsize)
    axes[1].set_ylabel('y [cm]', fontsize=fontsize)
    axes[1].set_title('Production position', fontsize=fontsize)
    axes[1].set_aspect('equal')
    fig.colorbar(im, ax=axes[1], label='density')

    pz_bins, pt_bins = np.linspace(0, 400, 201), np.linspace(0, 13, 201)
    pt = np.hypot(muons[:, 0], muons[:, 1])
    H, _, _ = np.histogram2d(muons[:, 2], pt, bins=[pz_bins, pt_bins], density=True)
    im = axes[2].pcolormesh(pz_bins, pt_bins, H.T, cmap='viridis', norm=LogNorm())
    axes[2].set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    axes[2].set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    axes[2].set_title('Momentum', fontsize=fontsize)
    fig.colorbar(im, ax=axes[2], label='density')

    fig.savefig(path)
    plt.close(fig)
    print(f'Saved plots to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configs = Path(__file__).resolve().parent / 'configs'
    parser.add_argument('model', nargs='?', default=str(configs / 'gmm_model.json'),
                        help='JSON file with the GMM parameters')
    parser.add_argument('--z_model', default=str(configs / 'z_beta_fit.json'),
                        help='JSON file with the Beta parameters of z')
    parser.add_argument('--n_samples', type=int, default=1_000_000, help='number of muons')
    parser.add_argument('--seed', type=int, default=None, help='random seed; None draws one')
    parser.add_argument('--save_data', action='store_true', help='output file (N, 7)')
    parser.add_argument('--plot', default='muons.png',
                        help="figure with the z, y vs x and pt vs pz histograms; '' to skip")
    args = parser.parse_args()

    muons = MuonsGenerator.from_file(args.model, args.z_model).sample(args.n_samples, args.seed).cpu().numpy()
    if args.save_data:
        np.save('muons.npy', muons)
        print(f'Saved {len(muons):,} muons to muons.npy')
    if args.plot:
        plot_muons(muons, args.plot)
