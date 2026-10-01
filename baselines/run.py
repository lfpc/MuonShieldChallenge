"""Run a black-box optimiser on a muon shield problem and save its history.

Every run first simulates the reference design, then the method until the budget of muons is spent: `budget` full
simulations, which an undersampling method (LCSO) spreads over more simulations of fewer muons each. Infeasible
designs are not simulated and cost nothing. The run is saved in <output_dir>/<config>.json, with the other runs.

Examples:
    python baselines/run.py cmaes configs/easy.json --budget 200 --seed 0
    python baselines/run.py turbo configs/hard.json --n_samples 1000000 --options '{"n_init": 20}'
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.evaluator import Evaluator, Exhausted  # noqa: E402
from baselines.methods import METHODS  # noqa: E402
from baselines.results import save_run  # noqa: E402
from muon_shield import MuonShieldProblem  # noqa: E402
from utils import load_config  # noqa: E402


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('config', help='problem configuration (JSON)')
    parser.add_argument('--budget', type=float, default=200,
                        help='muons to simulate, in full simulations (budget * n_samples muons), the reference included')
    parser.add_argument('--seed', type=int, default=0, help='seed of the optimiser')
    parser.add_argument('--options', default='{}', help='JSON dict of options of the method, see baselines/methods.py')
    parser.add_argument('--n_samples', type=int, default=None, help='override the number of muons of the config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--output_dir', default='results/baselines', help='where the runs are saved')
    args = parser.parse_args()

    config = load_config(args.config, n_samples=args.n_samples, output='n_hits')
    problem = MuonShieldProblem.from_config(config, device=f'cuda:{args.gpu}')
    ev = Evaluator(problem, args.budget)
    options = json.loads(args.options)
    print(f'{args.method} on {args.config} ({problem.dim} free parameters), budget {args.budget}, seed {args.seed}')

    t0 = time.time()
    try:
        ev(ev.x0)  # also loads the muons, which sets problem.n_samples used by the infeasibility penalty
        METHODS[args.method](ev, np.random.default_rng(args.seed), **options)
    except Exhausted:
        pass
    best = int(np.argmin(ev.F))
    print(f'Best f = {ev.F[best]:.6g} (reference {ev.F[0]:.6g}) after {len(ev.history)} simulations '
          f'({len(ev.F)} on all the muons), {ev.n_muons:.4g} muons, {ev.n_infeasible} infeasible proposals, '
          f'{time.time() - t0:.0f} s')

    path = Path(args.output_dir) / f'{Path(args.config).stem}.json'
    save_run(path, {'method': args.method, 'config': config, 'seed': args.seed, 'budget': args.budget,
                    'options': options, 'n_infeasible': ev.n_infeasible, 'n_muons': ev.n_muons, 'best_f': ev.F[best],
                    'best_phi': ev.to_phi(ev.U[best]).tolist(), 'history': ev.history})
    print(f'Saved {args.method}_seed{args.seed} in {path}')


if __name__ == '__main__':
    main()
