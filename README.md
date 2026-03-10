# Geometric Fisher Model — Evolutionary Simulation Framework

Fisher's Geometric Model (FGM) is one of the most influential theoretical
frameworks in evolutionary biology. It treats adaptation as a geometric
problem: a population occupies a cloud of points in an *n*-dimensional
phenotype space and must track a moving fitness optimum. This codebase
implements the model as a clean, extensible simulation that you can run,
inspect, and extend without modifying the core engine.

---

## The model

Every individual carries a phenotype vector **p** ∈ ℝⁿ. The environment
defines an optimal phenotype **α**(t) at each generation. Fitness is a
Gaussian function of phenotypic distance from that optimum:

```
φ(p, α) = exp( −‖p − α‖² / 2σ² )
```

A perfect match gives φ = 1; fitness decays exponentially with distance. The
width σ controls how strict the environment is — smaller σ means only
near-perfect phenotypes survive.

### Baseline scenario — "global warming"

The optimum drifts steadily in a fixed direction with small stochastic
fluctuations:

```
α(t+1) = α(t) + N(c, δ²I)
```

where **c** is the mean drift per generation and δ adds noise. The population
must continuously adapt or go extinct. Above a critical drift speed ‖c‖, no
population can keep up — this *critical drift threshold* is one of the key
quantities the model predicts.

### The evolutionary loop

Each generation runs four steps in order:

| Step | Operation | Implementation |
|------|-----------|----------------|
| 1 | **Mutation** — each phenotype is perturbed: trait *i* shifts by N(0, ξ²) with probability μ_c; the whole individual mutates with probability μ | `IsotropicMutation` |
| 2 | **Selection** — individuals with fitness below a threshold are removed; survivors are then re-sampled proportionally to fitness up to N | `TwoStageSelection` |
| 3 | **Reproduction** — survivors are drawn with replacement to restore population size N; each chosen individual produces one clone | `AsexualReproduction` |
| 4 | **Environment update** — α shifts by c + N(0, δ²I) | `LinearShiftEnvironment` |

> Mutation happens **before** selection, so it creates variation that
> selection can act on within the same generation — matching the standard
> FGM formulation.

### Fisher's dimensionality effect

A key prediction of FGM is that adaptation becomes harder as n grows. In
high-dimensional spaces, almost any random mutation is maladaptive regardless
of the current distance from the optimum, because most directions in ℝⁿ move
the phenotype *away* from it. This is why the mutation step size ξ must be
tuned carefully: too large and all mutations are harmful; too small and the
population cannot keep pace with the moving optimum.

---

## Repository structure

| File / directory | Role |
|------------------|------|
| `config.py` | All simulation parameters — **start here** |
| `strategies.py` | Abstract base classes defining the four extension interfaces |
| `main.py` | `run_simulation()` loop and GIF assembly |
| `individual.py` | `Individual` — holds a single phenotype vector |
| `population.py` | `Population` — container with initialisation logic |
| `mutation.py` | `IsotropicMutation` |
| `selection.py` | `TwoStageSelection`, `ThresholdSelection`, `ProportionalSelection` |
| `reproduction.py` | `AsexualReproduction` |
| `environment.py` | `LinearShiftEnvironment` |
| `stats.py` | `SimulationStats` — per-generation metrics, numpy array properties |
| `visualization.py` | GIF frame generation and summary plots |
| `run_experiment.py` | Parallel experiment runner — see below |
| `experiments/` | JSON experiment configs (one file = one reproducible condition) |
| `results/` | Generated output — **not committed**, lives only on your machine |

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

This runs 200 generations, saves PNG frames to `frames/`, assembles
`simulation.gif`, and opens a six-panel summary plot. All parameters live in
`config.py` — no other file needs to be touched for the baseline scenario.

### Example output

![Simulation GIF — baseline scenario (n=4, N=100, 200 generations)](simulation.gif)

*Baseline scenario: n = 4 trait dimensions, N = 100 individuals, 200 generations.
The fitness aura (green gradient) tracks the moving optimum (gold star).
Individuals are coloured from red (low fitness) to green (high fitness).
The fading white trail and arrow show the recent trajectory and predicted
next position of the optimum.*

---

## Key parameters (`config.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n` | 4 | Phenotype space dimensionality |
| `N` | 100 | Population size |
| `sigma` | 0.2 | Selection tolerance (smaller = stricter) |
| `xi` | 0.05 | Per-trait mutation step size |
| `mu` / `mu_c` | 0.1 / 0.5 | Mutation probabilities (per individual / per trait) |
| `c` | `[0.01, …]` | Mean optimum drift per generation |
| `delta` | 0.01 | Stochastic noise added to drift |
| `threshold` | 0.01 | Minimum fitness for survival (stage 1 of selection) |
| `init_scale` | 0.1 | Spread of initial phenotypes around α₀ |
| `seed` | 42 | RNG seed (`None` = new random result each run) |

> `alpha0` and `c` are derived from `n` automatically — changing `n` is
> safe, no manual vector resizing needed.

---

## Extending the framework

All four evolutionary steps are **pluggable**. To add a new mechanism:

1. **Subclass** the appropriate abstract class from `strategies.py`:
   `MutationStrategy`, `SelectionStrategy`, `ReproductionStrategy`, or
   `EnvironmentDynamics`
2. **Implement** the required method(s) — Python raises `TypeError`
   immediately if any abstract method is missing
3. **Pass** your instance to `run_simulation()` in `main.py`

Nothing else needs to change. Each extension lives in its own file.

### Adding a new statistic

Write into the `extra` dict on each `GenerationRecord`, then read it back:

```python
# write during your simulation (e.g. inside a SimulationStats subclass):
stats.records[-1].extra['my_metric'] = some_value

# read back as a numpy time series:
series = np.array([r.extra.get('my_metric', np.nan) for r in stats.records])
```

For a cleaner solution, subclass `SimulationStats` and override `record()`.
The docstring in `stats.py` shows the exact pattern.

---

## What the simulation produces

The **GIF** (`simulation.gif`) shows three panels per generation:
- *Left* — phenotype cloud (dimensions 1 and 2) overlaid on a Gaussian
  fitness aura, with a fading trail and forecast arrow showing the moving
  optimum; individuals coloured by their actual n-dimensional fitness
- *Centre* — mean fitness and distance from optimum over time
  (fixed x-axis spanning all generations for stable animation)
- *Right* — phenotypic variance (proxy for genetic diversity) and
  per-generation reproduction statistics

The **summary plot** (`plot_stats`) shows six panels: the three above plus,
when reproduction data is available, the number of "evolutionary winners"
(individuals with ≥ 1 offspring), the median offspring count among
reproducing individuals, and the maximum offspring count per generation.

---

## Running reproducible experiments

For collecting data across many replicates — which you need for any
statistical comparison — use the experiment runner:

```bash
python run_experiment.py experiments/baseline.json --workers 5
```

This runs 20 independent replicates in parallel (seeds 0–19, defined in the
JSON) and writes a timestamped directory to `results/`:

```
results/baseline_20260310_143022/
  config.json       — exact parameters used (copy of the input JSON)
  manifest.json     — git commit hash, Python version, OS, timestamp
  replicate_00.pkl  — full SimulationStats object (load with pickle)
  replicate_00.csv  — per-generation time series (open in Excel or pandas)
  ...
  summary.csv       — mean ± std across all replicates per generation
```

Each experiment config in `experiments/` is a self-contained JSON file
that fully specifies one condition — parameters, number of replicates, and
the exact seeds used. Three baseline configs are provided:

| Config | What it varies |
|--------|---------------|
| `baseline.json` | Reference condition (c = 0.01, n = 4) |
| `baseline_fast_drift.json` | Drift speed doubled (c = 0.02) |
| `baseline_high_dim.json` | Higher dimensionality (n = 8) |

To define your own experiment, copy one of these files, change the
parameters and `name` field, and run it. The `results/` directory is
excluded from git — data files stay on your machine; the config that
produced them stays in the repository.

Loading results for analysis:

```python
import pickle, pathlib, pandas as pd

run_dir = pathlib.Path('results/baseline_20260310_143022')
stats_list = [pickle.load(open(run_dir / f'replicate_{i:02d}.pkl', 'rb'))
              for i in range(20)]
summary = pd.read_csv(run_dir / 'summary.csv')
```

---

## Exploring results interactively

Once you have at least one completed experiment in `results/`, launch the
interactive viewer:

```bash
streamlit run viewer.py
```

Your browser opens automatically. The app has three pages, selectable from
the left sidebar:

### Overview page

A table of every run found in `results/`, showing the key parameters
(n, N, c, σ), number of replicates, how many went extinct, the final
mean fitness, and the exact git commit that produced the data — so every
row is fully traceable.

### Single run page

Pick any run to explore in depth:
- **Config** panel — the full JSON config used to produce the run
- **Manifest** panel — git commit hash, Python version, OS, and timestamp
- **Metric plots** — choose any combination of tracked metrics (mean
  fitness, distance from optimum, phenotypic variance, population size,
  reproduction statistics) and see mean ± std bands across all replicates
- **Individual replicate curves** — optional overlay of each replicate's
  raw time series; replicates that went extinct before the final generation
  are shown as dashed lines

### Compare two runs page

Select two conditions (e.g. `baseline` and `baseline_fast_drift`) from
the sidebar dropdowns. The app then shows:
- **Parameter diff table** — only the parameters that differ between the
  two configs, so the experimental contrast is immediately visible
- **Overlaid time-series** — all chosen metrics plotted together, with
  colour-coded mean ± std bands for each condition
- **Extinction rate chart** — bar chart of the fraction of replicates that
  went extinct, plus a Fisher's exact test p-value if each condition has
  at least 5 replicates
- **Final-generation snapshot** — a side-by-side table of mean ± std for
  every metric at the last recorded generation

> The viewer reads only the `summary.csv` and `config.json` / `manifest.json`
> files for the main views. Individual replicate data (`.pkl` / `.csv`) is
> loaded on demand when you tick "Show individual replicate curves".
