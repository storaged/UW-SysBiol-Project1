# Running experiments

This guide covers everything you need to collect reproducible, statistically
comparable data from the GFM simulation — from writing a config file to
loading the results in Python or a spreadsheet.

---

## Concepts

A *run* is one execution of `run_experiment.py` or `run_many_experiments.py`
for a single JSON config. Each run produces one timestamped directory under
`results/`. A *replicate* is a single evolutionary trajectory within that
run (different random seed, same parameters).

The separation between **configs** (in `experiments/`, committed to git) and
**results** (in `results/`, git-ignored) means every dataset is always
traceable: load a result directory, read `config.json` and `manifest.json`,
and you know exactly which parameters and code commit produced it.

---

## Running a single experiment

```bash
python run_experiment.py experiments/baseline.json
python run_experiment.py experiments/baseline.json --workers 4
```

`--workers N` sets the number of parallel processes. Defaults to
`n_replicates` (one worker per replicate). On a laptop with 8 cores, 4–6
workers is usually the sweet spot.

### Output layout

```
results/baseline_20260310_143022/
  config.json        — exact parameters (verbatim copy of the input JSON)
  manifest.json      — git commit hash, Python version, OS, timestamp
  replicate_00.pkl   — full SimulationStats object (pickle)
  replicate_00.csv   — per-generation time series (human-readable)
  replicate_01.pkl / .csv
  ...
  summary.csv        — mean ± std across all replicates per generation
```

`summary.csv` columns: `generation`, then for each metric
`<metric>_mean` / `<metric>_std`, plus `extinct_count` (how many
replicates had gone extinct by that generation).

---

## Running many experiments at once

```bash
python run_many_experiments.py <targets> [options]
```

`run_many_experiments.py` resolves one or more targets into a sorted,
deduplicated list of configs and runs them in sequence, printing a live
progress banner for each one.

### Target forms

| What you type | What it matches |
|---|---|
| `experiments/drift_exploration/` | Every `.json` in that subdirectory |
| `"experiments/drift_c*"` | Glob pattern — **quote** to prevent shell expansion |
| `--prefix drift_c` | Shorthand for `experiments/drift_c*.json` |
| `experiments/baseline.json` | A single specific file |

You can combine all forms freely:

```bash
# Run a whole subfolder plus a few extra files
python run_many_experiments.py experiments/my_sweep/ experiments/baseline.json

# Combine two prefixes
python run_many_experiments.py --prefix pop_N --prefix mut_xi

# Glob + individual file
python run_many_experiments.py "experiments/sel_*" experiments/baseline.json
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--workers N` | `n_replicates` | Parallel workers **per experiment** |
| `--yes` / `-y` | false | Skip the confirmation prompt |
| `--dry-run` | false | Print the resolved list and exit — nothing runs |
| `--stop-on-error` | false | Abort batch on first failure (default: continue) |

### Confirmation prompt

Before running, the script always prints a table of resolved configs with
replicate counts and descriptions, then asks for confirmation. Use `--dry-run`
to inspect the list without committing to run, and `--yes` to skip the prompt
in scripts or CI:

```bash
# Preview — shows the list, does not run
python run_many_experiments.py --prefix drift_c --dry-run

# Non-interactive — runs immediately
python run_many_experiments.py --prefix drift_c --yes --workers 4
```

---

## Writing a config file

Copy an existing file from `experiments/` and edit it. The minimal required
fields are:

```json
{
  "name":           "my_experiment",
  "description":    "One-line description shown in the viewer and dry-run output.",
  "group":          "My sweep",
  "n":              4,
  "N":              100,
  "sigma":          0.20,
  "xi":             0.05,
  "mu":             0.10,
  "mu_c":           0.50,
  "c":              0.01,
  "delta":          0.01,
  "threshold":      0.01,
  "init_scale":     0.10,
  "max_generations": 200,
  "n_replicates":   10,
  "seeds":          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

Key rules:

- `seeds` must have exactly `n_replicates` entries. Any integers work — by
  convention use `0, 1, 2, …, n_replicates - 1`.
- `c` can be a scalar (same drift in every dimension) or a list of length `n`
  for dimension-specific drift.
- `group` is optional but strongly recommended — the viewer uses it to colour
  charts and populate the "Filter by group" selector on the Parameter sweep
  page.

### Organising configs in subfolders

`experiments/` can have arbitrary subfolders:

```
experiments/
  baseline.json
  drift_exploration/
    drift_c0.005.json
    drift_c0.010.json
    drift_c0.015.json
  pop_size_sweep/
    pop_N25.json
    pop_N100.json
    pop_N200.json
```

Pass the subfolder name to `run_many_experiments.py`:

```bash
python run_many_experiments.py experiments/drift_exploration/ --workers 4
```

---

## Loading results in Python

```python
import pickle, pathlib, pandas as pd

run_dir = pathlib.Path('results/baseline_20260310_143022')

# Load a specific replicate
with open(run_dir / 'replicate_00.pkl', 'rb') as f:
    stats = pickle.load(f)

# Access per-generation records
for r in stats.records:
    print(r.generation, r.mean_fitness, r.population_size)

# Or use the numpy convenience properties
print(stats.mean_fitness_series)   # shape (n_generations,)
print(stats.distance_series)

# Load all replicates
n_reps = 20
all_stats = [
    pickle.load(open(run_dir / f'replicate_{i:02d}.pkl', 'rb'))
    for i in range(n_reps)
]

# Load the pre-computed summary (mean ± std across replicates)
summary = pd.read_csv(run_dir / 'summary.csv')
print(summary.columns.tolist())
```

The `SimulationStats` object also has:

```python
stats.extinct_at   # int or None — generation when population went extinct
stats.records      # list[GenerationRecord]
```

Each `GenerationRecord` has: `generation`, `mean_fitness`,
`distance_from_optimum`, `phenotype_variance`, `population_size`,
`n_parents`, `median_offspring`, `max_offspring`, and an `extra` dict
for any custom metrics you add.

---

## Reproducibility checklist

1. **Config committed** — every config in `experiments/` is tracked by git.
2. **Seeds explicit** — the `seeds` list in the JSON pins the RNG for every
   replicate; re-running the same config always gives the same numbers.
3. **Manifest recorded** — `manifest.json` in every result directory stores
   the git commit hash, Python version, and OS. If you need to reproduce a
   result exactly, check out that commit.
4. **Results are local** — `results/` is in `.gitignore`. Share configs, not
   data files.
