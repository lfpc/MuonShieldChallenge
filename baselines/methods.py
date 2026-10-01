"""Black-box optimisers. Each is a function (ev, rng, **options) that minimises ev(u) over u in [0, 1]^dim
until ev raises Exhausted. The reference design ev.x0 has already been simulated, and the initial designs
are drawn around it (spread sigma0), since almost no design drawn uniformly in the bounds is feasible.
"""
import numpy as np

from baselines.lcso import lcso


def random_search(ev, rng, sigma0=0.1):
    """Gaussian perturbations of the reference design."""
    while True:
        ev(ev.x0 + sigma0 * rng.standard_normal(ev.dim))


def around_x0(ev, rng, n, sigma0):
    """n designs around the reference design, the first being the reference design itself."""
    pop = np.clip(ev.x0 + sigma0 * rng.standard_normal((n, ev.dim)), 0, 1)
    pop[0] = ev.x0
    return pop


def genetic_algorithm(ev, rng, pop_size=20, n_elite=2, sigma0=0.1, mutation=0.1, blend=0.25):
    """Tournament selection (2), blend crossover (BLX-blend), Gaussian mutation of each gene with probability
    1/dim, elitism."""
    pop = around_x0(ev, rng, pop_size, sigma0)
    fit = np.array([ev(u) for u in pop])

    def select():
        i, j = rng.integers(pop_size, size=2)
        return pop[i] if fit[i] < fit[j] else pop[j]

    while True:
        elite = np.argsort(fit)[:n_elite]
        children = []
        for _ in range(pop_size - n_elite):
            a, b = select(), select()
            child = a + rng.uniform(-blend, 1 + blend, ev.dim) * (b - a)
            child += (rng.random(ev.dim) < 1 / ev.dim) * mutation * rng.standard_normal(ev.dim)
            children.append(np.clip(child, 0, 1))
        pop = np.vstack([pop[elite], children])
        fit = np.concatenate([fit[elite], [ev(u) for u in children]])


def differential_evolution(ev, rng, pop_size=20, F=0.5, CR=0.9, sigma0=0.1):
    """DE/rand/1/bin."""
    pop = around_x0(ev, rng, pop_size, sigma0)
    fit = np.array([ev(u) for u in pop])
    while True:
        for i in range(pop_size):
            a, b, c = pop[rng.choice(np.delete(np.arange(pop_size), i), 3, replace=False)]
            cross = rng.random(ev.dim) < CR
            cross[rng.integers(ev.dim)] = True
            trial = np.where(cross, np.clip(a + F * (b - c), 0, 1), pop[i])
            f = ev(trial)
            if f <= fit[i]:
                pop[i], fit[i] = trial, f


def cmaes(ev, rng, sigma0=0.1, popsize=None):
    """CMA-ES (pycma) started at the reference design."""
    import cma
    options = {'bounds': [0, 1], 'seed': int(rng.integers(1, 2**31)), 'verbose': -9}
    if popsize:
        options['popsize'] = popsize
    es = cma.CMAEvolutionStrategy(ev.x0, sigma0, options)
    while True:
        X = es.ask()
        es.tell(X, [ev(x) for x in X])


def bayesian_optimization(ev, rng, n_init=10, sigma0=0.1, n_candidates=2000, trust_region=False, length0=0.4):
    """GP (botorch SingleTaskGP) + LogEI, fitted on the simulated (feasible) designs only, on the problem's device.

    The acquisition is maximised over n_candidates feasible designs: perturbations of the best designs at several
    scales (or, with trust_region, TuRBO-1: points of a box around the best design whose side adapts to successes
    and failures, perturbing min(1, 20/dim) of the coordinates). The initial side length0 is smaller than TuRBO's
    0.8, as the search starts around the reference design.
    """
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    for u in around_x0(ev, rng, n_init, sigma0)[1:]:
        ev(u)
    length, n_success, n_fail = length0, 0, 0
    fail_tol = max(4, ev.dim)
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=ev.problem.device)
    while True:
        X, Y = tensor(ev.U), -tensor(ev.F).unsqueeze(1)  # botorch maximises
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        best = X[Y.argmax()].cpu().numpy()

        if trust_region:
            box = rng.uniform(-length / 2, length / 2, (n_candidates, ev.dim))
            mask = rng.random((n_candidates, ev.dim)) < min(1, 20 / ev.dim)
            cand = best + mask * box
        else:
            top = X[Y.squeeze(1).argsort(descending=True)[:5]].cpu().numpy()
            scales = np.array([0.01, 0.03, 0.1, 0.3])[rng.integers(4, size=(n_candidates, 1))]
            cand = top[rng.integers(len(top), size=n_candidates)] + scales * rng.standard_normal((n_candidates, ev.dim))
        cand = np.clip(cand, 0, 1)
        feasible = cand[[ev.feasible(c) for c in cand]]
        f_best = -Y.max().item()
        if len(feasible) == 0:
            u = cand[0]  # penalised, not simulated: a failure for the trust region
        else:
            with torch.no_grad():
                u = feasible[LogExpectedImprovement(gp, best_f=Y.max())(tensor(feasible).unsqueeze(1)).argmax().item()]
        f = ev(u)

        if trust_region:
            improved = f < f_best - 1e-3 * abs(f_best)
            n_success, n_fail = (n_success + 1, 0) if improved else (0, n_fail + 1)
            if n_success == 3:
                length, n_success = min(2 * length, 1.6), 0
            elif n_fail == fail_tol:
                length, n_fail = length / 2, 0
            if length < 0.5**7:  # restart the trust region
                length = length0


METHODS = {
    'random_search': random_search,
    'ga': genetic_algorithm,
    'de': differential_evolution,
    'cmaes': cmaes,
    'bo': bayesian_optimization,
    'turbo': lambda ev, rng, **kw: bayesian_optimization(ev, rng, trust_region=True, **kw),
    'lcso_gd': lcso,
    'lcso_newton': lambda ev, rng, **kw: lcso(ev, rng, newton=True, **kw),
}
