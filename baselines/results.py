"""The runs of a config, all in one JSON file {'<method>_seed<seed>': run}, shared by the runs of run_all.py."""
import fcntl
import json
from pathlib import Path


def load_runs(path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save_run(path, run: dict):
    """Add (or replace) a run. Under a lock, and written to a temporary file first, as runs finish in parallel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / f'.{path.name}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = load_runs(path)
        runs[f"{run['method']}_seed{run['seed']}"] = run
        tmp = path.with_name(f'.{path.name}.tmp')
        tmp.write_text(json.dumps(runs))
        tmp.replace(path)
