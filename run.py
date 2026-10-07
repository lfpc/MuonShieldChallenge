"""Simulate a muon shield design and compute its objective and constraint residuals.

Examples:
    python run.py configs/easy.json                      # reference design of the config
    python run.py configs/hard.json --phi phi_optm.txt   # design from a file (one value per line)
"""
import argparse
import json
import time

import numpy as np

from muon_shield import PARAM_NAMES, MuonShieldProblem
from utils import iron_cost, load_config, to_geometry, total_length


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument('config', help='problem configuration (JSON)')
    parser.add_argument('--phi', default=None,
                        help='text file with the free parameters or the full design, one value per line; '
                             'default is the reference design of the config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--n_samples', type=int, default=None, help='override the number of muons of the config')
    parser.add_argument('--seed', type=int, default=None, help='override the seed of the config')
    parser.add_argument('--output', default=None, help='save config, design and results to this JSON file')
    args = parser.parse_args()

    config = load_config(args.config, n_samples=args.n_samples, seed=args.seed, output='n_hits')
    problem = MuonShieldProblem.from_config(config, device=f'cuda:{args.gpu}')
    phi = problem.initial_phi if args.phi is None else np.loadtxt(args.phi)
    design = problem.design_from_phi(phi)

    print(f'Problem: {args.config} ({problem.dim} free parameters)')
    print('Design:')
    print('  ' + ' '.join(f'{name:>10}' for name in PARAM_NAMES))
    for row in design.tolist():
        print('  ' + ' '.join(f'{v:10.2f}' for v in row))

    t0 = time.time()
    n_hits = problem.objective(design)
    result = {'n_hits': n_hits, 'n_muons': problem.n_samples, 'length': total_length(design),
              'cost': iron_cost(to_geometry(design)), 'simulation_time': time.time() - t0,
              'constraints': problem.constraints(design).tolist()}
    print('\nResults:')
    for key, value in result.items():
        if key != 'constraints':
            print(f'  {key:<16} {value:.6g}')

    # Order of problem.constraints: length, cost, then 4 cavern overlaps per magnet if the cavern is simulated.
    g = result['constraints']
    print('Constraints (feasible iff all <= 0):')
    for name, value in zip(['length', 'cost'], g[:2]):
        print(f'  {name:<8} {value:.4g}' + ('  VIOLATED' if value > 0 else ''))
    if len(g) > 2:
        print('  cavern overlap (m)  ' + ' '.join(f'{side:>8}' for side in ('x_in', 'x_out', 'y_in', 'y_out')))
        for m, overlaps in enumerate(np.reshape(g[2:], (-1, 4))):
            print(f'    magnet {m}          ' + ' '.join(f'{v:8.3f}' for v in overlaps)
                  + ('  VIOLATED' if (overlaps > 0).any() else ''))

    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump({'config': config, 'design': design.tolist(), 'results': result}, f, indent=4)
        print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
