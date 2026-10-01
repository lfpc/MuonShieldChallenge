"""Local Classifier Surrogate Optimisation (LCSO) with the resample scheme of Robustness_MS (lcso_resample.py and
experiments/optb/lcso.py there).

Each round draws n_designs feasible designs in a box of half-width delta around the incumbent (the incumbent
first) and simulates each on its own random 1/resample_factor of the muons: many designs known roughly rather than
a few known precisely, as the design dependence is learnt across designs. A classifier of each muon's hit,
s(phi, x), is trained on a buffer of the last n_rounds_buffer rounds. Its predicted number of hits F(phi), the sum
over the incumbent's muons of sigmoid(s(phi, x)) scaled to all the muons, is the model of a trust-region method of
radius delta. The step minimises F's linear (lcso_gd) or quadratic (lcso_newton) model within the radius, the
bounds and the exact constraints (cheap, so not modelled), solved with SLSQP: it is feasible and can move along
active constraints. The candidate is simulated on all the muons, with rho = actual / predicted (by the model)
reduction of the hits: it is accepted if rho >= eta, delta is halved if rho < 0.25 and doubled (up to delta_max)
if rho > 0.75 and the step reached the boundary.

The classifier is Robustness_MS's TaylorBranchTrunk: quadratic in the design with a low-rank curvature, so its
design dependence is identifiable from few designs. Hits are rare, so each design keeps all its hits and
keep_negatives of its misses, reweighted to keep the loss (and F) unbiased.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn


class QuadraticClassifier(nn.Module):
    """Hit logit s(u, x) = a(x) + c(x)'v + 0.5 sum_j nu_j(x) (q_j'v)^2 + beta, with v = (u - u_loc) / u_scale the
    normalised design, a, c, nu an MLP of the normalised muon x and q_j an orthonormal basis of rank directions."""

    def __init__(self, dim, x_loc, x_scale, u_loc, u_scale, rank=8, hidden=128):
        super().__init__()
        self.dim, self.rank = dim, min(rank, dim)
        self.trunk = nn.Sequential(nn.Linear(len(x_loc), hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1 + dim + self.rank))
        self.Q_raw = nn.Parameter(0.5 * torch.randn(dim, self.rank))
        self.beta = nn.Parameter(torch.zeros(1))
        for name, value in (('x_loc', x_loc), ('x_scale', x_scale), ('u_loc', u_loc), ('u_scale', u_scale)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def coefficients(self, x):
        tau = self.trunk((x - self.x_loc) / self.x_scale)
        return tau[:, 0], tau[:, 1:1 + self.dim], tau[:, 1 + self.dim:], torch.linalg.qr(self.Q_raw)[0]

    def logit(self, u, x):
        """Logits of the pairs (u[i], x[i]): u (B, dim), x (B, 7)."""
        a, c, nu, Q = self.coefficients(x)
        v = (u - self.u_loc) / self.u_scale
        return a + (c * v).sum(-1) + 0.5 * (nu * (v @ Q) ** 2).sum(-1) + self.beta

    def predicted_hits(self, x, w):
        """F(u) = sum_i w_i sigmoid(s(u, x_i)) as a function of u (dim,), the MLP evaluated once."""
        with torch.no_grad():
            a, c, nu, Q = self.coefficients(x)
            beta = self.beta.detach()

        def F(u):
            v = (u - self.u_loc) / self.u_scale
            return (w * torch.sigmoid(a + c @ v + 0.5 * (nu * (v @ Q) ** 2).sum(-1) + beta)).sum()
        return F


def trust_region_step(ev, u, g, H, radius):
    """argmin of the model g'p + 0.5 p'Hp (H = 0: linear) over |p| <= radius, 0 <= u + p <= 1 and the exact
    constraints of u + p, by SLSQP from p = 0 (u is feasible). Returns p and the model's value at p. If SLSQP
    stops outside the radius (it can fail on an indefinite H), p is scaled back onto it; the caller checks the
    constraints."""
    model = lambda p: (g @ p + 0.5 * p @ H @ p, g + H @ p)
    constraints = [{'type': 'ineq', 'fun': lambda p: radius ** 2 - p @ p, 'jac': lambda p: -2 * p},
                   {'type': 'ineq', 'fun': lambda p: -ev.constraints(u + p)}]  # jacobian by finite differences
    p = minimize(model, np.zeros_like(u), jac=True, method='SLSQP', bounds=list(zip(-u, 1 - u)),
                 constraints=constraints, options={'maxiter': 200}).x
    p *= min(1, radius / max(np.linalg.norm(p), 1e-300))
    return p, model(p)[0]


def lcso(ev, rng, newton=False, n_designs=None, resample_factor=16, keep_negatives=20000, n_rounds_buffer=2,
         delta0=0.1, delta_max=0.5, eta=0.1, steps=200, batch=8192, lr=3e-3, rank=8, hidden=128):
    """LCSO until the budget is exhausted; see the module docstring. delta is in units of the unit cube."""
    device = ev.problem.device
    g = torch.Generator().manual_seed(int(rng.integers(2**31)))
    torch.manual_seed(int(rng.integers(2**31)))
    muons = ev.problem.muons
    pool = muons.to(device)
    N, dim = len(muons), ev.dim
    per_design, n_designs = max(1, N // resample_factor), n_designs or 2 * dim
    model = QuadraticClassifier(dim, pool.mean(0), pool.std(0).clamp_min(1e-6), ev.x0, delta0, rank, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    u_inc, f_inc, delta, buffer = ev.x0, ev.F[0], delta0, []

    while True:
        # designs around the incumbent, each simulated on its own random subset of the muons
        designs = [u_inc]
        for _ in range(100 * n_designs):
            if len(designs) == n_designs:
                break
            u = np.clip(u_inc + delta * rng.random() * rng.uniform(-1, 1, dim), 0, 1)  # radii spread from 0
            if ev.feasible(u):
                designs.append(u)
        for i, u in enumerate(designs):
            idx = torch.randperm(N, generator=g)[:per_design]
            hits = ev.simulate_subset(u, idx)
            if hits is None:
                continue
            neg = idx[~hits]
            kept = neg[torch.randperm(len(neg), generator=g)[:keep_negatives]]
            w_neg = len(neg) / max(len(kept), 1)
            y = torch.cat([torch.ones(int(hits.sum())), torch.zeros(len(kept))])
            w = torch.cat([torch.ones(int(hits.sum())), torch.full((len(kept),), w_neg)])
            buffer.append((tensor(u), torch.cat([idx[hits], kept]).to(device), y.to(device), w.to(device)))
            if i == 0:
                incumbent = buffer[-1]
        buffer = buffer[-n_rounds_buffer * n_designs:]

        # train the classifier on the buffer, continuing from the previous rounds
        U = torch.stack([b[0] for b in buffer])
        row_design = torch.cat([torch.full((len(b[1]),), j, device=device) for j, b in enumerate(buffer)])
        row_muon, row_y, row_w = (torch.cat([b[k] for b in buffer]) for k in (1, 2, 3))
        if len(buffer) == len(designs):  # first round: start from the mean hit rate (logit ~ -8), not from 1/2
            rate = float((row_w * row_y).sum() / row_w.sum())
            model.beta.data.fill_(np.log(max(rate, 1e-9) / (1 - rate)))
        model.train()
        for _ in range(steps):
            r = torch.randint(len(row_y), (batch,), device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(model.logit(U[row_design[r]], pool[row_muon[r]]),
                                                                  row_y[r], weight=row_w[r], reduction='sum')
            opt.zero_grad()
            (loss / row_w[r].sum()).backward()
            opt.step()
        model.eval()

        # step on the predicted hits of the incumbent's muons, scaled to all the muons
        F = model.predicted_hits(pool[incumbent[1]], incumbent[3] * N / per_design)
        ut = tensor(u_inc)
        grad = torch.func.grad(F)(ut).double().cpu().numpy()
        H = torch.func.hessian(F)(ut).double().cpu().numpy() if newton else np.zeros((dim, dim))
        p, m = trust_region_step(ev, u_inc, grad, 0.5 * (H + H.T), delta)
        for _ in range(10):  # SLSQP satisfies the constraints up to a tolerance: shorten the step if needed
            cand = np.clip(u_inc + p, 0, 1)
            if ev.feasible(cand):
                break
            p, m = p / 2, None
        if m is None:
            m = grad @ p + 0.5 * p @ H @ p
        f_cand = ev(cand)
        with torch.no_grad():
            f_pred = float(F(ut))
        predicted = -m
        actual, step = f_inc - f_cand, np.linalg.norm(cand - u_inc)
        rho = actual / predicted if predicted > 0 else -np.inf  # no predicted reduction: the model is wrong here
        print(f'  LCSO round: loss {loss.item() / row_w[r].sum().item():.3g}, f = {f_inc:.6g} (predicted {f_pred:.4g}) '
              f'-> {f_cand:.6g}, reduction {actual:.4g} (predicted {predicted:.4g}), rho {rho:.3g}, '
              f'step {step:.3g}, delta {delta:.3g}', flush=True)
        if rho >= eta:
            u_inc, f_inc = cand, f_cand
        if rho < 0.25:
            delta /= 2
        elif rho > 0.75 and step >= 0.99 * delta:
            delta = min(2 * delta, delta_max)
        if delta < delta0 / 40:  # re-expand rather than freeze at a tiny radius after noisy rounds
            delta = delta0 / 4
