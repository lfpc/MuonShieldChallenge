"""Compare the baselines saved by run.py: best f so far against the number of muons simulated (LCSO simulates most
of its designs on a fraction of the muons), mean over seeds with the min-max band, and a table of the final best f.

Example:
    python baselines/compare.py results/baselines/easy.json
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.results import load_runs  # noqa: E402


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def curve(run):
    """(muons simulated, best f so far) after each simulation of a run. Undersampled simulations have no f and
    keep the best f of the previous ones."""
    f = np.array([np.nan if h['f'] is None else h['f'] for h in run['history']])
    muons = np.cumsum([h.get('n_muons', run['config']['n_samples']) for h in run['history']])
    return muons, np.fmin.accumulate(f)


def compare(results, plot=None):
    """Print the table and save the figure (default: results with .png) of the runs in the JSON file results."""
    curves = defaultdict(list)
    for run in load_runs(results).values():
        curves[run['method']].append(curve(run))
    if not curves:
        raise SystemExit(f'No runs in {results}')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    print(f'{"method":<14} {"seeds":>5} {"muons":>10} {"best f (mean ± std)":>22} {"min":>10}')
    for method, runs in sorted(curves.items()):
        end = min(muons[-1] for muons, _ in runs)  # common range, if some runs stopped early
        grid = np.linspace(runs[0][0][0], end, 1000)
        best = np.array([best[np.searchsorted(muons, grid, side='right') - 1] for muons, best in runs])
        line, = ax.plot(grid, best.mean(0), label=f'{method} ({len(runs)})')
        ax.fill_between(grid, best.min(0), best.max(0), color=line.get_color(), alpha=0.2)
        final = np.array([best[-1] for _, best in runs])
        print(f'{method:<14} {len(runs):>5} {end:>10.3g} {final.mean():>12.6g} ± {final.std():<8.3g} {final.min():>10.6g}')
    ax.axhline(runs[0][1][0], color='k', ls='--', lw=1, label='reference design')
    ax.set(xlabel='muons simulated', ylabel='best number of hits so far', title=Path(results).stem)
    ax.legend()
    plot = plot or str(Path(results).with_suffix('.png'))
    fig.savefig(plot)
    print(f'Saved {plot}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument('results', help='JSON file with the runs of one config, e.g. results/baselines/easy.json')
    parser.add_argument('--plot', default=None, help='output figure; default the results file with .png')
    args = parser.parse_args()
    compare(args.results, args.plot)


if __name__ == '__main__':
    main()
