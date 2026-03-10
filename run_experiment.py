#!/usr/bin/env python3
"""
run_experiment.py — Reproducible parallel experiment runner.

Usage:
    python run_experiment.py experiments/baseline.json
    python run_experiment.py experiments/baseline.json --workers 4

Each run produces a timestamped output directory:

    results/
      <name>_<YYYYMMDD_HHMMSS>/
        config.json          — exact parameters used (copy of input config)
        manifest.json        — git commit hash, Python version, timestamp
        replicate_00.pkl     — full SimulationStats object (for further analysis)
        replicate_00.csv     — per-generation time series (human-readable)
        replicate_01.pkl / .csv
        ...
        summary.csv          — mean ± std across replicates per generation

Design principles
-----------------
  Reproducibility  — config.json + manifest.json travel with every result.
                     Anyone with the repo at the recorded commit can re-run
                     and get identical numbers.
  Separation       — experiment configs live in experiments/, results in
                     results/. Neither directory is the other's concern.
  Transparency     — .pkl files carry the full SimulationStats object;
                     .csv files are human-readable without Python.
  Extensibility    — adding a new strategy only requires updating the JSON
                     and the _build_strategies() helper below.

Loading results later
---------------------
    import pickle, pathlib
    run_dir = pathlib.Path('results/baseline_20260310_143022')
    stats_list = [pickle.load(open(run_dir / f'replicate_{i:02d}.pkl', 'rb'))
                  for i in range(5)]
    import pandas as pd
    summary = pd.read_csv(run_dir / 'summary.csv')
"""

import argparse
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Worker — must be a top-level function so multiprocessing can pickle it.
# All imports are inside the function body: on macOS Python uses 'spawn',
# meaning each worker process starts fresh and needs its own imports.
# ---------------------------------------------------------------------------

def _run_replicate(args: tuple) -> tuple:
    """
    Runs one simulation replicate.

    :param args: (cfg dict, seed int, replicate_index int)
    :return:     (replicate_index, SimulationStats)
    """
    cfg, seed, idx = args

    import numpy as np
    from population import Population
    from environment import LinearShiftEnvironment
    from selection import TwoStageSelection
    from reproduction import AsexualReproduction
    from mutation import IsotropicMutation
    from main import run_simulation

    np.random.seed(seed)

    n = cfg['n']
    alpha0 = np.zeros(n)

    # 'c' can be a scalar (same drift in every dimension) or a list
    c_raw = cfg.get('c', 0.01)
    c = np.full(n, c_raw) if np.isscalar(c_raw) else np.array(c_raw, dtype=float)

    pop = Population(cfg['N'], n, cfg['init_scale'], alpha_init=alpha0)
    env = LinearShiftEnvironment(alpha0.copy(), c.copy(), cfg.get('delta', 0.01))
    sel = TwoStageSelection(cfg['sigma'], cfg['threshold'], cfg['N'])
    rep = AsexualReproduction()
    mut = IsotropicMutation(cfg['mu'], cfg['mu_c'], cfg['xi'])

    stats = run_simulation(
        pop, env, sel, rep, mut,
        max_generations=cfg['max_generations'],
        frames_dir=None,
        verbose=False,
        target_size=cfg['N'],
        sigma=cfg['sigma'],
    )
    return idx, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Returns the short git commit hash of HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _stats_to_rows(stats) -> list:
    """Convert a SimulationStats object to a list of dicts for CSV export."""
    rows = []
    for r in stats.records:
        row = {
            'generation':            r.generation,
            'mean_fitness':          r.mean_fitness,
            'distance_from_optimum': r.distance_from_optimum,
            'phenotype_variance':    r.phenotype_variance,
            'population_size':       r.population_size,
            'n_parents':             r.n_parents,
            'median_offspring':      r.median_offspring,
            'max_offspring':         r.max_offspring,
            'extinct':               0,
        }
        # Include any student-defined extra metrics as extra_<key> columns
        for k, v in r.extra.items():
            row[f'extra_{k}'] = v
        rows.append(row)

    # Append an extinction marker row if the population went extinct
    if stats.extinct_at is not None:
        last_gen = rows[-1]['generation'] if rows else -1
        if last_gen < stats.extinct_at:
            rows.append({'generation': stats.extinct_at, 'extinct': 1})

    return rows


def _write_csv(rows: list, path: Path) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    # Collect all fieldnames in insertion order (extra columns may vary)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(all_stats: list, out_dir: Path) -> None:
    """
    Write summary.csv: per-generation mean ± std across all replicates.

    Columns: generation, <metric>_mean, <metric>_std, ..., extinct_count
    extinct_count is the number of replicates that went extinct at or before
    this generation — useful for survival-curve plots.
    """
    metrics = [
        'mean_fitness', 'distance_from_optimum', 'phenotype_variance',
        'population_size', 'n_parents', 'median_offspring', 'max_offspring',
    ]

    max_gen = max(
        (r.generation for s in all_stats for r in s.records),
        default=0,
    )

    rows = []
    for g in range(max_gen + 1):
        row = {'generation': g}
        values = {m: [] for m in metrics}
        extinct_count = 0

        for s in all_stats:
            if s.extinct_at is not None and s.extinct_at <= g:
                extinct_count += 1
                continue
            rec = next((r for r in s.records if r.generation == g), None)
            if rec is not None:
                for m in metrics:
                    values[m].append(getattr(rec, m))

        for m in metrics:
            vals = values[m]
            row[f'{m}_mean'] = float(np.mean(vals)) if vals else float('nan')
            row[f'{m}_std']  = float(np.std(vals))  if vals else float('nan')

        row['extinct_count'] = extinct_count
        rows.append(row)

    _write_csv(rows, out_dir / 'summary.csv')


# ---------------------------------------------------------------------------
# Public API  (importable by run_many_experiments.py and notebooks)
# ---------------------------------------------------------------------------

def run_one(config_path: Path | str, n_workers: int | None = None) -> Path:
    """
    Run all replicates for a single experiment config and return the output dir.

    :param config_path: Path to a JSON experiment config file.
    :param n_workers:   Number of parallel worker processes.
                        Defaults to min(n_replicates, CPU count).
    :return:            Path to the timestamped result directory.
    :raises SystemExit: If the config is missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")

    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)

    required = ['name', 'n', 'N', 'sigma', 'xi', 'mu', 'mu_c', 'c',
                'threshold', 'init_scale', 'max_generations', 'n_replicates', 'seeds']
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Error: config is missing required keys: {missing}")

    n_replicates = cfg['n_replicates']
    seeds = cfg['seeds']
    if len(seeds) != n_replicates:
        sys.exit(f"Error: 'seeds' length ({len(seeds)}) != 'n_replicates' ({n_replicates})")

    # ---- Create output directory ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('results') / f"{cfg['name']}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers_eff = min(n_workers or n_replicates, n_replicates)

    print(f"\nExperiment : {cfg['name']}")
    if cfg.get('description'):
        print(f"           : {cfg['description']}")
    print(f"Replicates : {n_replicates}  |  Workers: {n_workers_eff}")
    print(f"Output     : {out_dir}\n")

    # ---- Save config + manifest for full reproducibility ----
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    manifest = {
        'timestamp':      datetime.now().isoformat(timespec='seconds'),
        'git_commit':     _git_commit(),
        'python_version': sys.version.split()[0],
        'platform':       platform.platform(),
        'config_file':    str(config_path),
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # ---- Run replicates in parallel ----
    worker_args = [(cfg, seeds[i], i) for i in range(n_replicates)]
    all_stats = [None] * n_replicates

    with ProcessPoolExecutor(max_workers=n_workers_eff) as pool:
        futures = {pool.submit(_run_replicate, a): a[2] for a in worker_args}
        for fut in as_completed(futures):
            idx, stats = fut.result()
            all_stats[idx] = stats
            status   = 'EXTINCT' if stats.extinct_at is not None else 'survived'
            gen_done = (stats.extinct_at
                        if stats.extinct_at is not None
                        else (stats.records[-1].generation + 1 if stats.records else 0))
            print(f"  replicate {idx:02d}  seed={seeds[idx]:3d}  "
                  f"gens={gen_done:4d}  [{status}]")

    # ---- Save per-replicate files ----
    print()
    for i, stats in enumerate(all_stats):
        stem = f'replicate_{i:02d}'
        with open(out_dir / f'{stem}.pkl', 'wb') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_csv(_stats_to_rows(stats), out_dir / f'{stem}.csv')

    # ---- Save cross-replicate summary ----
    _write_summary(all_stats, out_dir)

    extinct = sum(1 for s in all_stats if s.extinct_at is not None)
    print(f"Saved      : {out_dir}")
    print(f"Files      : {n_replicates}× .pkl + .csv  |  summary.csv  |  "
          f"config.json  |  manifest.json")
    print(f"Extinct    : {extinct}/{n_replicates} replicates\n")

    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run parallel FGM replicates from a JSON experiment config.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python run_experiment.py experiments/baseline.json\n'
               '  python run_experiment.py experiments/baseline.json --workers 2',
    )
    parser.add_argument('config', help='Path to experiment JSON config file')
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel worker processes (default: n_replicates)',
    )
    args = parser.parse_args()

    # ---- Delegate to run_one() ----
    config_path = Path(args.config)
    run_one(config_path, n_workers=args.workers)


if __name__ == '__main__':
    main()
