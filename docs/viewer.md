# Interactive results viewer

The viewer is a [Streamlit](https://streamlit.io) web app that lets you
explore experiment results without writing any Python. Launch it with:

```bash
streamlit run viewer.py
```

Your default browser opens automatically at `http://localhost:8501`. The app
re-scans the `results/` directory on every page load; use the
**🔄 Refresh results** button in the sidebar after running a new experiment.

---

## Sidebar controls

| Control | Purpose |
|---|---|
| **🔄 Refresh results** | Re-scans `results/` to pick up newly completed runs |
| **Show all re-runs** | By default, only the most recent result for each experiment name is shown. Enable this toggle to see every re-run individually |
| **Page** radio | Switches between the four pages described below |

---

## Page 1 — Overview

A bird's-eye view of everything in `results/`.

**Group chips** at the top count how many conditions and total replicates
belong to each experiment group (e.g. "Drift speed sweep", "Population size
sweep"). Groups are taken from the `group` field in `config.json`; if it is
absent the viewer infers the group from the run name.

**Extinction rate chart** — a colour-coded bar chart (one bar per condition,
colour = group) showing the fraction of replicates that went extinct. The 50 %
line is drawn for reference.

**All conditions table** — one row per run showing group, key parameters
(c, n, N, σ), replicate count, extinction count and percentage, final mean
fitness, timestamp, and short git hash.

---

## Page 2 — Single run

Deep-dive into one experiment.

### Selecting a run

Use the **Select run** dropdown in the sidebar. Runs are labelled
`[Group] name` and sorted by group then by the numeric part of the name, so
related conditions appear together.

### Config and manifest panels

Side-by-side JSON viewers showing the exact parameters and provenance
(git commit, Python version, OS, timestamp) of the selected run.

### Summary statistics

Choose any combination of tracked metrics from the multiselect box:

| Metric | What it measures |
|---|---|
| Mean fitness | Average $e^{-\|p-\alpha\|^2/2\sigma^2}$ across the population |
| Distance from optimum | Mean Euclidean distance $\|p - \alpha\|$ |
| Phenotype variance | Trace of the phenotype covariance matrix — a proxy for genetic diversity |
| Population size | Number of individuals alive |
| Parents (n_parents) | Number of individuals that produced at least one offspring |
| Median offspring | Median offspring count among reproducing individuals |
| Max offspring | Largest number of offspring from a single individual |

Each metric is plotted as a **mean ± std band** across all replicates. The
shaded region is ± 1 standard deviation; the solid line is the mean.

### Individual replicate curves

Tick **Show individual replicate curves** to overlay the raw time series of
every replicate. Replicates that went extinct before the final generation are
drawn as dashed lines so you can see which trajectories ended early.

### Extinct replicates counter

A large metric card at the bottom shows `extinct / n_replicates` at a glance.

### 🎬 Phenotype-space GIF

At the bottom of the Single run page you can generate an animated GIF
on demand:

1. Pick a **random seed** (0 to `n_replicates − 1`). Each seed gives a
   different evolutionary trajectory with the same parameters.
2. Adjust the **frame duration** slider (0.05 – 0.50 s per frame; 0.15 is
   a good default).
3. Click **▶ Generate GIF**. The simulation reruns from scratch and assembles
   the animation — this takes about as long as one replicate.

The GIF is displayed inline and loops automatically. Click **↩ Restart from
generation 0** to jump back to the first frame at any time. Previously
generated GIFs are cached: if you navigate away and come back with the same
seed the cached file is shown without rerunning. Use **⬇ Download GIF** to
save the animation locally.

The GIF shows three panels per generation:
- *Left* — phenotype cloud (dimensions 1 and 2) on a Gaussian fitness aura;
  individuals coloured by their actual n-dimensional fitness (red → green);
  fading white trail and forecast arrow for the moving optimum
- *Centre* — mean fitness and distance from optimum over time
- *Right* — phenotypic variance and reproduction statistics

---

## Page 3 — Compare two runs

Select **Condition A** and **Condition B** from the two sidebar dropdowns.

**Parameter diff table** — shows only the parameters that differ between the
two configs, making the experimental contrast immediately visible.

**Overlaid time-series** — choose metrics with the multiselect, then see
colour-coded mean ± std bands for both conditions on the same axes.

**Extinction rate chart** — a two-bar chart with the fraction of extinct
replicates per condition. If each condition has at least 5 replicates, a
Fisher's exact test p-value is displayed.

**Final-generation snapshot** — a side-by-side table of mean ± std for every
metric at the last recorded generation.

---

## Page 4 — Parameter sweep

Designed for comparing many conditions that vary a single parameter. Use the
**Filter by group** selector in the sidebar to narrow the candidate runs to
one experiment group (e.g. "Drift speed sweep"), then use the
**Runs to include** multiselect to pick the exact conditions you want.

The page has five tabs:

### ⚙️ Setup

Before diving into plots, this tab shows you exactly what you are comparing:

- **Fixed parameters** — parameters that are the same across all selected runs
- **Varied parameters** — parameters that differ, with the range of values
- **Per-condition parameter cards** — colour-coded tiles, one per run, listing
  every parameter value
- **Provenance table** — timestamp, git commit, and replicate count for each
  run

### 🌡️ Landscape heatmap

A heat map with conditions on the y-axis and generations on the x-axis. Each
cell is coloured by the chosen metric, making it easy to spot at a glance
where and when conditions diverge. Extinct replicates are hatched.

### 📈 Dose-response

Plots the final-generation value of the chosen metric against the swept
parameter. Each point is the cross-replicate mean; error bars are ± 1 std.
Useful for locating critical thresholds (e.g. the critical drift speed).

### 🔀 Trajectories

Overlaid mean ± std time-series for all selected conditions, colour-coded by a
sweep palette. Equivalent to Page 3 but for many conditions at once.

### 🔬 Adaptive dynamics

A derived-quantity panel:

- **Lag** — mean distance from optimum at the final generation vs the swept
  parameter
- **Speed of adaptation** — first generation where mean fitness rises above a
  chosen threshold (default 0.5)
- **Extinction probability** — fraction of replicates extinct at the final
  generation
- **Diversity decay** — phenotypic variance at the final generation vs the
  swept parameter

---

## Tips

- **Group naming** — add a `"group"` key to every config JSON. The viewer
  uses it for chart colours, table chips, and the sweep page filter. If the
  key is absent the viewer infers the group from the run name prefix.
- **Re-running experiments** — by default the sidebar only shows the most
  recent result for each experiment name. Enable **Show all re-runs** to see
  every timestamped result directory individually.
- **Large result sets** — the viewer loads `summary.csv` for all runs on the
  Overview page. Individual `.pkl` / `.csv` files are only loaded on demand
  (when you select a run or tick "Show individual replicate curves"), so the
  app stays fast even with many experiments.
