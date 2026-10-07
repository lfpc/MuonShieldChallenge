"""Budgeted unit-cube interface to a MuonShieldProblem, shared by every baseline."""
import time

import numpy as np


class Exhausted(Exception):
    """Raised by Evaluator when the budget of muons (or the cap on infeasible proposals) is used up."""


class Evaluator:
    """The optimisers work on u in [0, 1]^dim, with phi = low + u (high - low).

    The budget counts simulated muons, in units of full simulations: budget * n_samples muons in total. A full
    simulation (__call__) uses all n_samples muons, an undersampled one (simulate_subset) only those it is given.

    The constraints are cheap, so an infeasible design is not simulated: it gets f = n_samples * (1 + violation),
    worse than any feasible design (at most n_samples hits), and costs nothing. The violation is the sum of the
    positive residuals, length and cost divided by L0 and W0, cavern overlaps in m.
    """

    def __init__(self, problem, budget: float, max_infeasible_ratio: int = 100):
        self.problem, self.budget = problem, budget
        self.max_infeasible = int(max_infeasible_ratio * budget)
        self.low, self.high = problem.bounds.numpy()
        self.dim = problem.dim
        self.x0 = ((problem.initial_phi.numpy() - self.low) / (self.high - self.low)).clip(0, 1)
        n_g = len(problem.constraints(problem.initial_phi))
        self.g_scale = np.array([problem.L0, problem.W0] + [1.0] * (n_g - 2))
        self.U, self.F, self.history, self.n_infeasible, self.n_muons = [], [], [], 0, 0

    @property
    def max_muons(self) -> float:
        return self.budget * self.problem.n_samples

    def to_phi(self, u) -> np.ndarray:
        return self.low + np.clip(u, 0, 1) * (self.high - self.low)

    def constraints(self, u) -> np.ndarray:
        """Constraint residuals of u, feasible if all <= 0: length / L0, cost / W0, cavern overlaps (m)."""
        return self.problem.constraints(self.to_phi(u)).numpy() / self.g_scale

    def violation(self, u) -> float:
        return float(np.maximum(self.constraints(u), 0).sum())

    def feasible(self, u) -> bool:
        return self.violation(u) == 0

    def _check(self, u, n_muons):
        """u clipped to the cube, or None (counted) if infeasible; Exhausted if the budget cannot pay n_muons."""
        if self.n_muons + n_muons > self.max_muons or self.n_infeasible >= self.max_infeasible:
            raise Exhausted
        u = np.clip(np.asarray(u, dtype=np.float64), 0, 1)
        if self.feasible(u):
            return u
        self.n_infeasible += 1
        return None

    def _record(self, u, f, n_muons, t0):
        self.n_muons += n_muons
        self.history.append({'f': f, 'n_muons': n_muons} if f is None else  # undersampled: only its muons
                            {'phi': self.to_phi(u).tolist(), 'f': f, 'n_muons': n_muons, 'time': time.time() - t0,
                             'n_infeasible': self.n_infeasible})

    def __call__(self, u) -> float:
        """f of u on all the muons: the simulated number of hits if feasible, else the penalty above."""
        v = self.violation(u)
        u = self._check(u, self.problem.n_samples)
        if u is None:
            return self.problem.n_samples * (1 + v)
        t0 = time.time()
        f = float(self.problem.objective(self.to_phi(u)))
        self.U.append(u)
        self.F.append(f)
        self._record(u, f, self.problem.n_samples, t0)
        print(f'  simulation {len(self.history)}: f = {f:.6g}, best = {min(self.F):.6g}, '
              f'muons {self.n_muons / self.max_muons:.1%} of the budget', flush=True)
        return f

    def charge(self, n_muons):
        """Count n_muons simulated outside the evaluator (e.g. RL_opt's partial shields) towards the budget."""
        if self.n_muons + n_muons > self.max_muons:
            raise Exhausted
        self._record(None, None, n_muons, time.time())

    def simulate_subset(self, u, idx):
        """Hits (bool, one per muon) of design u simulated on the muons idx of the problem's sample only, or None
        if u is infeasible (not simulated). In the history, these undersampled simulations only have their muons."""
        u = self._check(u, len(idx))
        if u is None:
            return None
        t0 = time.time()
        muons = self.problem.muons
        self.problem._muons = muons[idx]  # simulate these muons only
        try:
            hits = self.problem.hits(self.problem.simulate(self.to_phi(u)))
        finally:
            self.problem._muons = muons
        self._record(u, None, len(idx), t0)
        return hits
