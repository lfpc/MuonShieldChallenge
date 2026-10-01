"""Run every baseline n_repeats times (seeds 0, ..., n_repeats - 1), spread over the GPUs (one run at a time per
GPU), then plot the comparison.

Each run is a baselines/run.py process. Arguments not listed below are passed to every run (e.g. --n_samples).
Ctrl-C (or kill) stops all the runs. Runs: <output_dir>/<config>.json, plot: <output_dir>/<config>.png, logs (for
debugging): <tmp>/muonshield_baselines/<config>/<method>_seed<seed>.log.

Examples:
    python baselines/run_all.py configs/easy.json --budget 200 --gpus 0 1 2 3 --n_repeats 3
    python baselines/run_all.py configs/hard.json --methods cmaes turbo --n_samples 1000000
"""
import argparse
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.compare import compare  # noqa: E402
from baselines.methods import METHODS  # noqa: E402


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument('config', nargs='?', default=str(ROOT / 'configs' / 'easy.json'), help='problem configuration')
    parser.add_argument('--budget', type=float, default=400, help='muons per run, in full simulations (see run.py)')
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS, help='methods to run')
    parser.add_argument('--n_repeats', type=int, default=5, help='runs of each method, with seeds 0, ..., n_repeats - 1')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU indices, one run at a time on each')
    parser.add_argument('--output_dir', default=str(ROOT / 'results' / 'baselines'), help='where the runs are saved')
    args, run_args = parser.parse_known_args()

    name = Path(args.config).stem
    logs = Path(tempfile.gettempdir()) / 'muonshield_baselines' / name
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in args.methods for s in range(args.n_repeats)]
    print(f'{len(jobs)} runs ({len(args.methods)} methods x {args.n_repeats} repeats) on GPUs {args.gpus}, logs in {logs}')

    procs, lock, stop = [], threading.Lock(), threading.Event()

    def worker(i, gpu):
        for m, s in jobs[i::len(args.gpus)]:
            log = logs / f'{m}_seed{s}.log'
            with lock:
                if stop.is_set():
                    return
                print(f'GPU {gpu}: {m} seed {s}', flush=True)
                with open(log, 'w') as f:
                    p = subprocess.Popen([sys.executable, str(ROOT / 'baselines' / 'run.py'), m, args.config,
                                          '--budget', str(args.budget), '--seed', str(s), '--gpu', str(gpu),
                                          '--output_dir', args.output_dir, *run_args],
                                         stdout=f, stderr=subprocess.STDOUT)
                procs.append(p)
            if p.wait() != 0 and not stop.is_set():
                error = [line for line in log.read_text(errors='replace').splitlines() if line.strip()][-1:]
                print(f'FAILED: {m} seed {s}: {"".join(error)} (log: {log})', flush=True)

    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)  # also when started in the background, where SIGINT is ignored
    signal.signal(signal.SIGTERM, interrupt)
    threads = [threading.Thread(target=worker, args=(i, gpu), daemon=True) for i, gpu in enumerate(args.gpus)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        with lock:
            stop.set()
            for p in procs:
                p.terminate()
        raise SystemExit('Stopped all runs')
    compare(Path(args.output_dir) / f'{name}.json')


if __name__ == '__main__':
    main()
