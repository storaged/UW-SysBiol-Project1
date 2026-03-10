"""
viewer.py — Interactive results explorer for the GFM experiment runner.

Launch:
    streamlit run viewer.py

The app scans the results/ directory for completed experiment runs (produced
by run_experiment.py), lets the user select up to two conditions, and shows:

  Page 1 – Overview      : list of all runs with key metadata
  Page 2 – Single run    : replicate-level detail (individual CSV files)
  Page 3 – Compare two   : overlaid time-series with mean ± std bands,
                           config diff, extinction rates, reproduction stats
"""

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GFM Experiment Viewer",
    page_icon="🧬",
    layout="wide",
)

RESULTS_DIR = Path("results")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_group(name: str) -> str:
    """Infer an experiment group from the run name when no 'group' key exists
    in config.json (e.g. for results produced before the field was introduced)."""
    if name.startswith("drift_c"):
        return "Drift speed sweep"
    if "baseline" in name:
        return "Baseline conditions"
    if name.startswith("pop_") or "small_pop" in name or "large_pop" in name:
        return "Population size sweep"
    if name.startswith("mut_"):
        return "Mutation step sweep"
    if name.startswith("sel_"):
        return "Selection tolerance sweep"
    return "Other"


def build_run_options(
    runs: list[dict], dedup: bool = True
) -> tuple[list[str], list[dict]]:
    """Sort and optionally deduplicate runs for UI display.

    Deduplication — when the same config was run more than once, keep only the
    most recent result directory.  ``discover_runs`` returns dirs newest-first
    so the first occurrence of each name wins.

    Sorting — groups sorted alphabetically; within each group runs are sorted
    by the trailing numeric value in their name (e.g. drift_c0.006 < 0.014),
    falling back to alphabetical.

    Returns ``(display_labels, ordered_runs)`` — 1-to-1 aligned lists.
    Labels are formatted as  ``"[Group]  name"``  for Streamlit widgets.
    """
    if dedup:
        seen: dict[str, dict] = {}
        for r in runs:
            if r["name"] not in seen:
                seen[r["name"]] = r
        working = list(seen.values())
    else:
        working = list(runs)

    def _sort_key(r: dict):
        g = r.get("group", "Other")
        m = re.search(r"(\d+\.?\d*)$", r["name"])
        num = float(m.group(1)) if m else 0.0
        return (g, num, r["name"])

    working.sort(key=_sort_key)
    display_labels = [f"[{r.get('group', 'Other')}]  {r['name']}" for r in working]
    return display_labels, working


def discover_runs(results_dir: Path) -> list[dict]:
    """Scan results/ for subdirectories containing config.json + summary.csv.
    Returns a list of dicts, sorted newest-first.

    **Not cached** — the scan is cheap and this ensures new result directories
    appear immediately without requiring a manual app restart.
    """
    runs = []
    if not results_dir.exists():
        return runs
    for d in sorted(results_dir.iterdir(), reverse=True):
        cfg_path  = d / "config.json"
        sum_path  = d / "summary.csv"
        mani_path = d / "manifest.json"
        if not (d.is_dir() and cfg_path.exists() and sum_path.exists()):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        manifest = {}
        if mani_path.exists():
            with open(mani_path, encoding="utf-8") as f:
                manifest = json.load(f)
        _name = cfg.get("name", d.name)
        runs.append({
            "dir":       d,
            "label":     d.name,
            "name":      _name,
            "group":     cfg.get("group") or _auto_group(_name),
            "cfg":       cfg,
            "manifest":  manifest,
            "n_reps":    cfg.get("n_replicates", "?"),
            "timestamp": manifest.get("timestamp", ""),
            "git":       manifest.get("git_commit", "?"),
        })
    return runs


@st.cache_data
def load_summary(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "summary.csv")


@st.cache_data
def load_replicate_csvs(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all replicate_NN.csv files from a run directory."""
    files = sorted(run_dir.glob("replicate_*.csv"))
    return {i: pd.read_csv(f) for i, f in enumerate(files)}


def cfg_diff_table(cfg_a: dict, cfg_b: dict) -> pd.DataFrame:
    """Return a DataFrame showing only parameters that differ between two configs."""
    all_keys = sorted(set(cfg_a) | set(cfg_b))
    rows = []
    for k in all_keys:
        va = cfg_a.get(k, "—")
        vb = cfg_b.get(k, "—")
        if va != vb:
            rows.append({"parameter": k, "Condition A": va, "Condition B": vb})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["parameter", "Condition A", "Condition B"]
    )


def ts_plot(ax, df: pd.DataFrame, col: str, label: str, color: str,
            alpha_fill: float = 0.15) -> None:
    """Plot mean ± std time series on the given Axes."""
    mean_col = f"{col}_mean"
    std_col  = f"{col}_std"
    if mean_col not in df.columns:
        return
    gens = df["generation"]
    mean = df[mean_col]
    std  = df[std_col] if std_col in df.columns else pd.Series(np.zeros(len(df)))
    ax.plot(gens, mean, color=color, lw=1.8, label=label)
    ax.fill_between(gens, mean - std, mean + std,
                    color=color, alpha=alpha_fill)


METRIC_LABELS = {
    "mean_fitness":          ("Mean fitness φ",          (0, 1)),
    "distance_from_optimum": ("Distance from optimum",   None),
    "phenotype_variance":    ("Phenotypic variance",      None),
    "population_size":       ("Population size",          None),
    "n_parents":             ("Evolutionary winners (≥1 offspring)", None),
    "median_offspring":      ("Median offspring (among reproducing)", None),
    "max_offspring":         ("Max offspring",             None),
}

COLORS = {"A": "#2196F3", "B": "#E91E63"}   # blue / pink

PARAM_LABELS: dict[str, str] = {
    "c":               "drift speed",
    "n":               "phenotype dimensions",
    "N":               "population size",
    "sigma":           "selection width σ",
    "xi":              "mutation step ξ",
    "mu":              "per-individual mutation prob.",
    "mu_c":            "per-trait mutation prob.",
    "delta":           "drift noise δ",
    "threshold":       "survival threshold",
    "init_scale":      "initial phenotype spread",
    "max_generations": "max generations",
}

# ── Helpers for parameter sweep ───────────────────────────────────────────────

def detect_swept_params(selected_runs: list[dict]) -> list[str]:
    """Return numeric config keys that vary across the selected runs."""
    if len(selected_runs) < 2:
        return []
    all_keys = sorted(set().union(*[set(r["cfg"].keys()) for r in selected_runs]))
    result = []
    for k in all_keys:
        vals = [r["cfg"].get(k) for r in selected_runs]
        if all(isinstance(v, (int, float)) for v in vals if v is not None):
            if len({v for v in vals if v is not None}) > 1:
                result.append(k)
    return result


def sweep_palette(n: int, cmap_name: str = "plasma") -> list:
    """Return n colours evenly spaced along a matplotlib colormap."""
    cmap_obj = plt.get_cmap(cmap_name)
    return [cmap_obj(i / max(n - 1, 1)) for i in range(n)]


@st.cache_data
def build_ts_matrix(
    run_dir_strs: tuple[str, ...],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build matrix[n_runs × n_gens] from summary CSVs.
    Missing / post-extinction cells → NaN.
    """
    dfs = [pd.read_csv(Path(d) / "summary.csv") for d in run_dir_strs]
    all_gens_set: set[int] = set()
    for df in dfs:
        all_gens_set.update(df["generation"].tolist())
    all_gens = np.array(sorted(all_gens_set))
    col = f"{metric}_mean"
    matrix = np.full((len(dfs), len(all_gens)), np.nan)
    for i, df in enumerate(dfs):
        vals = df[col].tolist() if col in df.columns else [np.nan] * len(df)
        g2v = dict(zip(df["generation"].tolist(), vals))
        for j, g in enumerate(all_gens):
            matrix[i, j] = g2v.get(int(g), np.nan)
    return matrix, all_gens


def first_gen_above(df: pd.DataFrame, col: str, threshold: float) -> int | None:
    """Return the first generation where col >= threshold, else None."""
    if col not in df.columns:
        return None
    hits = df[df[col] >= threshold]
    return int(hits["generation"].iloc[0]) if not hits.empty else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🧬 GFM Viewer")
st.sidebar.markdown("---")

if st.sidebar.button(
    "🔄 Refresh results",
    help="Re-scan the results/ directory. Use after running a new experiment.",
):
    st.rerun()

runs = discover_runs(RESULTS_DIR)

if not runs:
    st.title("No experiment results found")
    st.markdown(
        f"Run an experiment first:\n```bash\n"
        f"python run_experiment.py experiments/baseline.json --workers 5\n```\n"
        f"Results will appear in **`{RESULTS_DIR}/`**."
    )
    st.stop()

show_reruns = st.sidebar.checkbox(
    "Show all re-runs",
    value=False,
    help=(
        "By default only the most recent result directory for each experiment "
        "name is shown.  Enable to reveal every re-run individually."
    ),
)
display_labels, ordered_runs = build_run_options(runs, dedup=not show_reruns)
run_by_display = {lbl: r for lbl, r in zip(display_labels, ordered_runs)}

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Single run", "Compare two runs", "Parameter sweep"],
)

# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Experiment overview")

    # ── Group summary chips ───────────────────────────────────────────────────
    groups: dict[str, list[dict]] = {}
    for r in ordered_runs:
        groups.setdefault(r.get("group", "Other"), []).append(r)

    grp_names = sorted(groups.keys())
    grp_cols  = st.columns(len(grp_names))
    for ci, grp_name in enumerate(grp_names):
        grp_runs = groups[grp_name]
        n_reps_t = sum(r["n_reps"] for r in grp_runs if isinstance(r["n_reps"], int))
        grp_cols[ci].metric(
            grp_name,
            f"{len(grp_runs)} condition(s)",
            f"{n_reps_t} total replicates",
        )

    st.markdown("---")

    # ── Extinction-rate overview chart ────────────────────────────────────────
    with st.expander("📊 Extinction rate across all conditions", expanded=True):
        ext_rows = []
        for r in ordered_runs:
            s   = load_summary(r["dir"])
            ext = int(s["extinct_count"].max()) if "extinct_count" in s.columns else 0
            n_r = r["n_reps"] if isinstance(r["n_reps"], int) else 1
            ext_rows.append({
                "name":  r["name"],
                "group": r.get("group", "Other"),
                "rate":  ext / n_r * 100,
            })
        ext_df    = pd.DataFrame(ext_rows)
        uniq_grps = sorted(ext_df["group"].unique())
        tab10     = plt.get_cmap("tab10")
        g_color   = {g: tab10(i / 10.0) for i, g in enumerate(uniq_grps)}
        bar_clrs  = [g_color[row["group"]] for _, row in ext_df.iterrows()]

        fig_ov, ax_ov = plt.subplots(figsize=(max(9, len(ext_df) * 0.62), 4.2))
        ax_ov.bar(range(len(ext_df)), ext_df["rate"],
                  color=bar_clrs, edgecolor="white", linewidth=0.5)
        ax_ov.set_xticks(range(len(ext_df)))
        ax_ov.set_xticklabels(ext_df["name"], rotation=40, ha="right", fontsize=8)
        ax_ov.set_ylabel("Extinction rate (%)")
        ax_ov.set_ylim(0, 118)
        ax_ov.axhline(50, ls="--", color="#888", lw=1.2)
        ax_ov.text(
            len(ext_df) - 0.5, 53, "50 %",
            fontsize=7.5, color="#666", va="bottom", ha="right",
        )
        legend_handles = [
            mpatches.Patch(color=g_color[g], label=g) for g in uniq_grps
        ]
        ax_ov.legend(handles=legend_handles, fontsize=8, loc="upper left")
        ax_ov.set_title(
            "Extinction rate by condition  (colour = experiment group)", fontsize=11
        )
        plt.tight_layout()
        st.pyplot(fig_ov)
        plt.close()

    # ── Full conditions table ─────────────────────────────────────────────────
    st.subheader("All conditions")
    rows = []
    for r in ordered_runs:
        s        = load_summary(r["dir"])
        ext      = int(s["extinct_count"].max()) if "extinct_count" in s.columns else "?"
        n_r      = r["n_reps"]
        last_fit = (
            float(s["mean_fitness_mean"].iloc[-1])
            if "mean_fitness_mean" in s.columns else float("nan")
        )
        rows.append({
            "Group":         r.get("group", "Other"),
            "Name":          r["name"],
            "c":             r["cfg"].get("c",     "?"),
            "n":             r["cfg"].get("n",     "?"),
            "N":             r["cfg"].get("N",     "?"),
            "σ":             r["cfg"].get("sigma", "?"),
            "Replicates":    n_r,
            "Extinct":       ext,
            "Extinct %":     (
                f"{ext/n_r*100:.0f}%"
                if isinstance(ext, int) and isinstance(n_r, int) else "?"
            ),
            "Final fitness": f"{last_fit:.4f}" if not np.isnan(last_fit) else "n/a",
            "Timestamp":     r["timestamp"][:19] if r["timestamp"] else "",
            "Git":           r["git"][:7] if r["git"] not in ("?", "") else "?",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        f"Showing {len(ordered_runs)} condition(s) from `{RESULTS_DIR}/`. "
        "Toggle **Show all re-runs** in the sidebar to include earlier runs of "
        "the same experiment.  "
        "Add more: `python run_experiment.py experiments/<config>.json`."
    )

# ── Page: Single run ─────────────────────────────────────────────────────────

elif page == "Single run":
    st.title("Single run detail")

    sel = st.sidebar.selectbox("Select run", display_labels)
    run = run_by_display[sel]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Config")
        st.json(run["cfg"])
    with col2:
        st.subheader("Manifest (provenance)")
        st.json(run["manifest"])

    summary = load_summary(run["dir"])
    rep_csvs = load_replicate_csvs(run["dir"])

    st.markdown("---")
    st.subheader("Summary statistics (mean ± std across replicates)")

    metrics_available = [m for m in METRIC_LABELS if f"{m}_mean" in summary.columns]
    chosen = st.multiselect(
        "Metrics to show",
        options=metrics_available,
        default=[m for m in ["mean_fitness", "distance_from_optimum",
                              "phenotype_variance", "n_parents"] if m in metrics_available],
        format_func=lambda m: METRIC_LABELS[m][0],
    )

    if chosen:
        n_cols = min(2, len(chosen))
        n_rows = (len(chosen) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        for i, metric in enumerate(chosen):
            ax = axes[i // n_cols][i % n_cols]
            ts_plot(ax, summary, metric, run["name"], COLORS["A"])
            label, ylim = METRIC_LABELS[metric]
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Generation")
            if ylim:
                ax.set_ylim(*ylim)
        for j in range(len(chosen), n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Individual replicates")

    show_reps = st.checkbox("Show individual replicate curves", value=False)
    rep_metric = st.selectbox(
        "Metric",
        options=[m for m in METRIC_LABELS if m in (rep_csvs[0].columns if rep_csvs else [])],
        format_func=lambda m: METRIC_LABELS[m][0],
    ) if rep_csvs else None

    if show_reps and rep_csvs and rep_metric:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, df in rep_csvs.items():
            if rep_metric in df.columns:
                extinct = df["extinct"].max() if "extinct" in df.columns else 0
                style = "--" if extinct else "-"
                ax.plot(df["generation"], df[rep_metric],
                        color=COLORS["A"], alpha=0.35, lw=1, linestyle=style)
        # overlay mean
        if f"{rep_metric}_mean" in summary.columns:
            ax.plot(summary["generation"], summary[f"{rep_metric}_mean"],
                    color="black", lw=2, label="mean")
        ax.set_title(METRIC_LABELS[rep_metric][0])
        ax.set_xlabel("Generation")
        ax.legend()
        ylim = METRIC_LABELS[rep_metric][1]
        if ylim:
            ax.set_ylim(*ylim)
        st.pyplot(fig)
        plt.close()
        st.caption("Dashed lines = replicates that went extinct before the final generation.")

    if "extinct_count" in summary.columns:
        max_ext = int(summary["extinct_count"].max())
        n_reps  = run["cfg"].get("n_replicates", "?")
        st.metric("Extinct replicates", f"{max_ext} / {n_reps}")

# ── Page: Compare two runs ────────────────────────────────────────────────────

elif page == "Compare two runs":
    st.title("Compare two conditions")

    sel_a = st.sidebar.selectbox("Condition A", display_labels, index=0)
    sel_b = st.sidebar.selectbox(
        "Condition B",
        display_labels,
        index=min(1, len(display_labels) - 1),
    )

    run_a = run_by_display[sel_a]
    run_b = run_by_display[sel_b]

    sum_a = load_summary(run_a["dir"])
    sum_b = load_summary(run_b["dir"])

    # ── Config diff ──────────────────────────────────────────────────────────
    st.subheader("Parameter differences")
    diff = cfg_diff_table(run_a["cfg"], run_b["cfg"])
    if diff.empty:
        st.info("The two configs have identical parameters.")
    else:
        diff = diff.rename(columns={
            "Condition A": f"A — {run_a['name']}",
            "Condition B": f"B — {run_b['name']}",
        })
        st.dataframe(diff, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Time-series comparison ────────────────────────────────────────────────
    st.subheader("Time-series comparison (mean ± std across replicates)")

    metrics_available = [
        m for m in METRIC_LABELS
        if f"{m}_mean" in sum_a.columns or f"{m}_mean" in sum_b.columns
    ]
    chosen = st.multiselect(
        "Metrics to plot",
        options=metrics_available,
        default=[m for m in ["mean_fitness", "distance_from_optimum",
                              "phenotype_variance", "n_parents"] if m in metrics_available],
        format_func=lambda m: METRIC_LABELS[m][0],
    )

    if chosen:
        n_cols = min(2, len(chosen))
        n_rows = (len(chosen) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 3.5 * n_rows),
                                 squeeze=False)
        for i, metric in enumerate(chosen):
            ax = axes[i // n_cols][i % n_cols]
            ts_plot(ax, sum_a, metric, run_a["name"], COLORS["A"])
            ts_plot(ax, sum_b, metric, run_b["name"], COLORS["B"])
            label, ylim = METRIC_LABELS[metric]
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Generation")
            if ylim:
                ax.set_ylim(*ylim)
            patch_a = mpatches.Patch(color=COLORS["A"], label=f'A — {run_a["name"]}')
            patch_b = mpatches.Patch(color=COLORS["B"], label=f'B — {run_b["name"]}')
            ax.legend(handles=[patch_a, patch_b], fontsize=8)
        for j in range(len(chosen), n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Extinction bar chart ─────────────────────────────────────────────────
    st.subheader("Extinction rates")

    ext_a = int(sum_a["extinct_count"].max()) if "extinct_count" in sum_a.columns else 0
    ext_b = int(sum_b["extinct_count"].max()) if "extinct_count" in sum_b.columns else 0
    n_a   = run_a["cfg"].get("n_replicates", 1)
    n_b   = run_b["cfg"].get("n_replicates", 1)

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(
        [f'A — {run_a["name"]}', f'B — {run_b["name"]}'],
        [ext_a / n_a * 100, ext_b / n_b * 100],
        color=[COLORS["A"], COLORS["B"]],
        width=0.4,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Extinction rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Fraction of replicates that went extinct")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    col1.metric(f"A — {run_a['name']} extinct", f"{ext_a} / {n_a}")
    col2.metric(f"B — {run_b['name']} extinct", f"{ext_b} / {n_b}")

    if n_a >= 5 and n_b >= 5:
        try:
            from scipy.stats import fisher_exact
            table = [[ext_a, n_a - ext_a], [ext_b, n_b - ext_b]]
            _, p = fisher_exact(table)
            st.caption(
                f"Fisher's exact test: p = {p:.4f} "
                f"({'significant' if p < 0.05 else 'not significant'} at α = 0.05)"
            )
        except ImportError:
            st.caption("Install scipy for automatic Fisher's exact test on extinction rates.")

    st.markdown("---")

    # ── Final-generation snapshot ────────────────────────────────────────────
    st.subheader("Final generation snapshot")

    snap_metrics = [m for m in METRIC_LABELS if f"{m}_mean" in sum_a.columns]
    rows = []
    for m in snap_metrics:
        va_m = sum_a[f"{m}_mean"].iloc[-1]
        va_s = sum_a[f"{m}_std"].iloc[-1]  if f"{m}_std"  in sum_a.columns else float("nan")
        vb_m = sum_b[f"{m}_mean"].iloc[-1]
        vb_s = sum_b[f"{m}_std"].iloc[-1]  if f"{m}_std"  in sum_b.columns else float("nan")
        rows.append({
            "Metric": METRIC_LABELS[m][0],
            f"A — {run_a['name']}  (mean ± std)": f"{va_m:.4f} ± {va_s:.4f}",
            f"B — {run_b['name']}  (mean ± std)": f"{vb_m:.4f} ± {vb_s:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "Values computed from the last generation present in summary.csv. "
        "Extinct replicates are excluded from the mean and std."
    )

# ── Page: Parameter sweep ─────────────────────────────────────────────────────

elif page == "Parameter sweep":
    st.title("Parameter sweep explorer")
    st.markdown(
        "Select runs that vary **one parameter** to reveal how the model responds "
        "across a range of conditions. The app auto-detects the swept axis and "
        "builds four complementary visualisations."
    )

    # ── Group filter (narrows the multiselect below) ─────────────────
    all_groups = sorted({r.get("group", "Other") for r in ordered_runs})
    group_filter = st.sidebar.selectbox(
        "Filter by group",
        ["— all groups —"] + all_groups,
        help="Pick a group to pre-fill only those runs in the selector below.",
    )

    if group_filter == "— all groups —":
        candidate_labels = display_labels
    else:
        candidate_labels = [
            lbl for lbl in display_labels
            if run_by_display[lbl].get("group", "Other") == group_filter
        ]

    sel_sweep = st.sidebar.multiselect(
        "Runs to include",
        candidate_labels,
        default=candidate_labels,
        help="Fine-tune which runs from the selected group to compare.",
    )

    if len(sel_sweep) < 2:
        st.info(
            "Select at least 2 runs from the sidebar to start.  "
            "Choose a **group** above to quickly load an entire sweep series."
        )
        st.stop()

    sweep_runs = [run_by_display[lbl] for lbl in sel_sweep]
    swept_params = detect_swept_params(sweep_runs)
    if not swept_params:
        st.warning(
            "No numeric parameter differs across the selected runs. "
            "Try selecting runs from a sweep series (e.g., different drift speeds c)."
        )
        st.stop()

    param_choice = st.sidebar.selectbox(
        "Sweep axis",
        swept_params,
        format_func=lambda p: f"{p}  —  {PARAM_LABELS.get(p, p)}",
    )

    # Sort runs by parameter value
    sorted_runs = sorted(sweep_runs, key=lambda r: r["cfg"].get(param_choice, 0))
    param_vals  = [r["cfg"].get(param_choice) for r in sorted_runs]
    run_names   = [r["name"] for r in sorted_runs]
    n_runs      = len(sorted_runs)
    param_label = PARAM_LABELS.get(param_choice, param_choice)

    # Continuous colormap mapped to parameter values
    norm       = mcolors.Normalize(vmin=min(param_vals), vmax=max(param_vals))
    cmap_sweep = plt.get_cmap("plasma")
    colors     = [cmap_sweep(norm(v)) for v in param_vals]
    sm         = plt.cm.ScalarMappable(cmap=cmap_sweep, norm=norm)
    sm.set_array([])

    # ── Summary chips ────────────────────────────────────────────────────────
    chip_cols = st.columns(n_runs)
    for ci, (run, val, col) in enumerate(zip(sorted_runs, param_vals, colors)):
        hex_c = mcolors.to_hex(col)
        r_v, g_v, b_v = col[:3]
        text_color = "#111111" if (0.299 * r_v + 0.587 * g_v + 0.114 * b_v) > 0.55 else "#ffffff"
        chip_cols[ci].markdown(
            f"<div style='background:{hex_c};padding:6px 4px;border-radius:6px;"
            f"text-align:center;color:{text_color};font-size:12px'>"
            f"<b>{param_choice}={val}</b><br>"
            f"<span style='font-size:10px'>{run['name']}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Shared data pre-computed before tabs
    summaries      = {r["label"]: load_summary(r["dir"]) for r in sorted_runs}
    global_max_gen = max(int(summaries[r["label"]]["generation"].max()) for r in sorted_runs)
    metrics_available = [
        m for m in METRIC_LABELS
        if any(f"{m}_mean" in summaries[r["label"]].columns for r in sorted_runs)
    ]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Setup",
        "🌡️ Landscape",
        "📈 Dose-response",
        "🔀 Trajectories",
        "🔬 Adaptive dynamics",
    ])

    # ── Tab 0: Setup ─────────────────────────────────────────────────────────
    with tab0:
        st.subheader("Experimental setup")
        st.markdown(
            "This tab documents exactly which parameters are **held constant** "
            "across all selected conditions and which ones **vary** — making the "
            "experimental design transparent and reproducible."
        )

        # Build the fixed / varied split
        all_cfg_keys = sorted(
            {k for r in sorted_runs for k in r["cfg"]
             if k not in ("name", "description", "group", "seeds")}
        )
        fixed, varied = {}, {}
        for k in all_cfg_keys:
            vals_k = [r["cfg"].get(k) for r in sorted_runs]
            if len({str(v) for v in vals_k}) == 1:
                fixed[k] = vals_k[0]
            else:
                varied[k] = vals_k

        col_fix, col_var = st.columns([1, 1], gap="large")

        with col_fix:
            st.markdown("### 🔒 Fixed parameters")
            st.caption("Same value in every condition of this sweep.")
            fix_rows = [
                {"Parameter": k,
                 "Label": PARAM_LABELS.get(k, k),
                 "Value": str(v)}
                for k, v in fixed.items()
            ]
            st.dataframe(
                pd.DataFrame(fix_rows),
                use_container_width=True,
                hide_index=True,
            )

        with col_var:
            st.markdown("### 🔀 Varied parameters")
            st.caption("These differ across conditions — the experimental axes.")
            var_rows = []
            for k, vals_k in varied.items():
                var_rows.append({
                    "Parameter": k,
                    "Label": PARAM_LABELS.get(k, k),
                    "Min": min(v for v in vals_k if v is not None),
                    "Max": max(v for v in vals_k if v is not None),
                    "# levels": len(set(vals_k)),
                    "Values": ", ".join(str(v) for v in vals_k),
                })
            st.dataframe(
                pd.DataFrame(var_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # Per-condition parameter cards
        st.markdown("### 📋 Per-condition parameter cards")
        st.caption(
            "Each card shows the swept parameter value in the header and lists "
            "every fixed parameter below it for quick reference."
        )
        card_cols = st.columns(min(n_runs, 5))
        for ci, (run, val, col) in enumerate(zip(sorted_runs, param_vals, colors)):
            hex_c     = mcolors.to_hex(col)
            r_v, g_v, b_v = col[:3]
            text_c    = "#111" if (0.299*r_v + 0.587*g_v + 0.114*b_v) > 0.55 else "#fff"
            ext_count = 0
            if "extinct_count" in summaries[run["label"]].columns:
                ext_count = int(summaries[run["label"]]["extinct_count"].max())
            n_reps = run["cfg"].get("n_replicates", "?")
            ext_pct = f"{ext_count/n_reps*100:.0f}%" if isinstance(n_reps, int) else "?"

            fixed_lines = "".join(
                f"<tr><td style='padding:1px 6px;color:#555;font-size:10px'>{k}</td>"
                f"<td style='padding:1px 4px;font-size:10px'>{run['cfg'].get(k,'—')}</td></tr>"
                for k in sorted(fixed)
            )
            card_cols[ci % 5].markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;"
                f"overflow:hidden;margin-bottom:8px'>"
                f"<div style='background:{hex_c};padding:8px 10px;color:{text_c}'>"
                f"<div style='font-size:13px;font-weight:700'>{param_choice} = {val}</div>"
                f"<div style='font-size:10px;opacity:.85'>{run['name']}</div>"
                f"<div style='font-size:10px;margin-top:2px'>extinct {ext_count}/{n_reps} ({ext_pct})</div>"
                f"</div>"
                f"<div style='padding:6px 4px'>"
                f"<table style='width:100%;border-collapse:collapse'>{fixed_lines}</table>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Provenance table
        st.markdown("### 🔬 Provenance")
        st.caption("Git commit and timestamp for every result directory in this sweep.")
        prov_rows = [
            {
                "Condition":   r["name"],
                f"{param_choice}": r["cfg"].get(param_choice),
                "Result dir":  r["dir"].name,
                "Timestamp":   r["timestamp"][:19] if r["timestamp"] else "—",
                "Git commit":  r["git"][:7] if r["git"] not in ("?", "") else "?",
                "Python":      r["manifest"].get("python_version", "—"),
                "OS":          r["manifest"].get("platform", "—"),
            }
            for r in sorted_runs
        ]
        st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)
    # ── Tab 1: Landscape heatmap ──────────────────────────────────────────────
    with tab1:
        st.subheader("Metric landscape over time")
        st.markdown(
            "Each row is one condition sorted by the swept parameter. "
            "Colour encodes the metric value. **Grey cells = extinction** "
            "(population did not survive to that generation)."
        )
        h_metric = st.selectbox(
            "Metric",
            metrics_available,
            format_func=lambda m: METRIC_LABELS[m][0],
            key="hmap_metric",
        )
        run_dir_strs = tuple(str(r["dir"]) for r in sorted_runs)
        matrix, gens = build_ts_matrix(run_dir_strs, h_metric)

        hmap_cmap_name = "RdYlGn" if h_metric == "mean_fitness" else "viridis"
        fig, ax = plt.subplots(figsize=(13, max(2.5, 0.75 * n_runs)))
        masked = np.ma.masked_invalid(matrix)
        used_cmap = plt.get_cmap(hmap_cmap_name).copy()
        used_cmap.set_bad(color="#cccccc")
        im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=used_cmap)
        plt.colorbar(im, ax=ax, label=METRIC_LABELS[h_metric][0], fraction=0.03, pad=0.02)
        n_xticks  = min(10, len(gens))
        xtick_idx = np.linspace(0, len(gens) - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([str(int(gens[i])) for i in xtick_idx])
        ax.set_xlabel("Generation")
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(
            [f"{param_choice}={v}  ({n})" for v, n in zip(param_vals, run_names)],
            fontsize=9,
        )
        ax.set_title(
            f"{METRIC_LABELS[h_metric][0]} — sweep over {param_label}",
            fontsize=12,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Dose-response ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Dose-response at a chosen generation")
        st.markdown(
            "Drag the slider to watch how the dose-response curve **evolves over time**. "
            "Error bars show ± 1 SD across surviving replicates."
        )
        d_metric = st.selectbox(
            "Metric",
            metrics_available,
            format_func=lambda m: METRIC_LABELS[m][0],
            key="dose_metric",
        )
        gen_choice = st.slider(
            "Generation",
            min_value=1,
            max_value=global_max_gen,
            value=global_max_gen,
            key="dose_gen",
        )

        means_at_gen, stds_at_gen, extinct_rates = [], [], []
        for r in sorted_runs:
            df_s = summaries[r["label"]]
            df_row = df_s[df_s["generation"] <= gen_choice].tail(1)
            mcol, scol = f"{d_metric}_mean", f"{d_metric}_std"
            if not df_row.empty and mcol in df_row.columns:
                means_at_gen.append(float(df_row[mcol].iloc[0]))
                stds_at_gen.append(float(df_row[scol].iloc[0]) if scol in df_row.columns else 0.0)
            else:
                means_at_gen.append(np.nan)
                stds_at_gen.append(0.0)
            n_reps = r["cfg"].get("n_replicates", 1)
            ext = int(df_s["extinct_count"].max()) if "extinct_count" in df_s.columns else 0
            extinct_rates.append(ext / n_reps * 100)

        fig, (ax_dose, ax_ext) = plt.subplots(
            2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1.5]}
        )
        valid_x = [x for x, y in zip(param_vals, means_at_gen) if not np.isnan(y)]
        valid_y = [y for y in means_at_gen if not np.isnan(y)]
        valid_e = [e for e, y in zip(stds_at_gen, means_at_gen) if not np.isnan(y)]
        if valid_x:
            ax_dose.plot(valid_x, valid_y, color="#555", lw=1.5, zorder=1)
            ax_dose.errorbar(
                valid_x, valid_y, yerr=valid_e,
                fmt="none", ecolor="#999", capsize=5, capthick=1.2, zorder=2,
            )
        for xv, yv, col in zip(param_vals, means_at_gen, colors):
            if not np.isnan(yv):
                ax_dose.scatter([xv], [yv], color=col, s=110, zorder=3,
                                edgecolors="k", linewidths=0.8)
        ax_dose.set_xlabel(f"{param_choice}  ({param_label})", fontsize=11)
        ax_dose.set_ylabel(METRIC_LABELS[d_metric][0], fontsize=11)
        ax_dose.set_title(
            f"{METRIC_LABELS[d_metric][0]} at generation {gen_choice}", fontsize=11
        )
        ylim = METRIC_LABELS[d_metric][1]
        if ylim:
            ax_dose.set_ylim(*ylim)
        ax_dose.grid(True, alpha=0.3)

        bars = ax_ext.bar(
            [str(v) for v in param_vals], extinct_rates,
            color=colors, edgecolor="k", linewidth=0.6,
        )
        ax_ext.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)
        ax_ext.set_xlabel(f"{param_choice}")
        ax_ext.set_ylabel("Extinction (%)")
        ax_ext.set_ylim(0, 115)
        ax_ext.set_title("Extinction rate (final generation)", fontsize=10)
        ax_ext.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Overlaid trajectories ──────────────────────────────────────────
    with tab3:
        st.subheader("Multi-condition time series")
        st.markdown(
            "Lines are coloured along the **plasma** colormap from low (dark) to "
            f"high (bright) values of *{param_choice}*. "
            "Shaded bands show ± 1 SD across replicates."
        )
        traj_metrics = st.multiselect(
            "Metrics to plot",
            metrics_available,
            default=[m for m in ["mean_fitness", "distance_from_optimum",
                                  "phenotype_variance", "population_size"]
                     if m in metrics_available],
            format_func=lambda m: METRIC_LABELS[m][0],
            key="traj_metrics",
        )
        if traj_metrics:
            n_m  = len(traj_metrics)
            n_mc = min(2, n_m)
            n_mr = (n_m + 1) // 2
            fig, axes = plt.subplots(
                n_mr, n_mc,
                figsize=(7 * n_mc, 3.8 * n_mr),
                squeeze=False,
            )
            for mi, metric in enumerate(traj_metrics):
                ax = axes[mi // n_mc][mi % n_mc]
                for run, val, col in zip(sorted_runs, param_vals, colors):
                    df_s = summaries[run["label"]]
                    mcol, scol = f"{metric}_mean", f"{metric}_std"
                    if mcol not in df_s.columns:
                        continue
                    gns  = df_s["generation"]
                    mean = df_s[mcol]
                    std  = df_s[scol] if scol in df_s.columns else 0
                    ax.plot(gns, mean, color=col, lw=1.8,
                            label=f"{param_choice}={val}")
                    ax.fill_between(gns, mean - std, mean + std,
                                    color=col, alpha=0.12)
                label_str, ylim = METRIC_LABELS[metric]
                ax.set_title(label_str, fontsize=10)
                ax.set_xlabel("Generation")
                if ylim:
                    ax.set_ylim(*ylim)
            for j in range(n_m, n_mr * n_mc):
                axes[j // n_mc][j % n_mc].set_visible(False)
            fig.subplots_adjust(right=0.87)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
            cb = fig.colorbar(sm, cax=cbar_ax)
            cb.set_label(f"{param_choice}  ({param_label})", fontsize=9)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Select at least one metric above.")

    # ── Tab 4: Adaptive dynamics ──────────────────────────────────────────────
    with tab4:
        st.subheader("Adaptive dynamics")
        st.markdown(
            "Four panels that together describe *when* and *how well* populations "
            "adapt as the swept parameter changes."
        )
        adapt_threshold = st.slider(
            "Adaptation threshold (mean fitness)",
            min_value=0.1, max_value=0.95, value=0.6, step=0.05,
            key="adapt_thresh",
            help="'Adaptation time' shows the first generation at which mean "
                 "fitness reached this value.",
        )

        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.38)
        ax_phase = fig.add_subplot(gs[0, 0])
        ax_adapt = fig.add_subplot(gs[0, 1])
        ax_corr  = fig.add_subplot(gs[1, 0])
        ax_var   = fig.add_subplot(gs[1, 1])

        adapt_gens, final_fitness, final_var = [], [], []
        for run, val, col in zip(sorted_runs, param_vals, colors):
            df_s = summaries[run["label"]]

            # Panel A: phase portrait trajectory
            if "mean_fitness_mean" in df_s.columns and "distance_from_optimum_mean" in df_s.columns:
                xs = df_s["distance_from_optimum_mean"]
                ys = df_s["mean_fitness_mean"]
                ax_phase.plot(xs, ys, color=col, lw=1.6, alpha=0.85)
                ax_phase.scatter([xs.iloc[0]],  [ys.iloc[0]],
                                 color=col, s=40, marker="o", zorder=4)
                ax_phase.scatter([xs.iloc[-1]], [ys.iloc[-1]],
                                 color=col, s=90, marker="D", zorder=5,
                                 edgecolors="k", linewidths=0.7)

            adapt_gens.append(
                first_gen_above(df_s, "mean_fitness_mean", adapt_threshold)
            )
            final_fitness.append(
                df_s["mean_fitness_mean"].iloc[-1]
                if "mean_fitness_mean" in df_s.columns else np.nan
            )
            final_var.append(
                df_s["phenotype_variance_mean"].iloc[-1]
                if "phenotype_variance_mean" in df_s.columns else np.nan
            )

        ax_phase.set_xlabel("Distance from optimum")
        ax_phase.set_ylabel("Mean fitness φ")
        ax_phase.set_title("Phase portrait\n(○ = gen 0,  ◇ = final gen)", fontsize=10)
        ax_phase.set_xlim(left=0)
        ax_phase.set_ylim(0, 1)
        fig.colorbar(sm, ax=ax_phase, fraction=0.055, pad=0.04).set_label(
            f"{param_choice}", fontsize=8
        )

        # Panel B: adaptation time
        bar_heights = [ag if ag is not None else global_max_gen * 1.08
                       for ag in adapt_gens]
        bars_ad = ax_adapt.bar(
            [str(v) for v in param_vals], bar_heights,
            color=colors, edgecolor="k", linewidth=0.6,
        )
        for i, ag in enumerate(adapt_gens):
            if ag is None:
                ax_adapt.text(
                    i, bar_heights[i] * 0.96, "never",
                    ha="center", va="top", fontsize=8,
                    color="white", fontweight="bold",
                )
        ax_adapt.axhline(global_max_gen, ls="--", color="#888", lw=1, label="max gen")
        ax_adapt.set_xlabel(f"{param_choice}")
        ax_adapt.set_ylabel("Generation")
        ax_adapt.set_title(
            f"Adaptation time\n(first gen with fitness ≥ {adapt_threshold:.2f})",
            fontsize=10,
        )
        ax_adapt.legend(fontsize=8)

        # Panels C & D share valid (non-NaN) data
        valid_mask = [not np.isnan(f) for f in final_fitness]
        pv_v = [v  for v, m in zip(param_vals,    valid_mask) if m]
        ff_v = [f  for f, m in zip(final_fitness, valid_mask) if m]
        fv_v = [f  for f, m in zip(final_var,     valid_mask) if m]
        cv_v = [c  for c, m in zip(colors,         valid_mask) if m]

        # Panel C: final fitness vs parameter
        ax_corr.scatter(pv_v, ff_v, c=cv_v, s=110, edgecolors="k",
                        linewidths=0.8, zorder=3)
        if len(pv_v) >= 2:
            slope, intercept = np.polyfit(pv_v, ff_v, 1)
            xs_fit = np.linspace(min(pv_v), max(pv_v), 50)
            ax_corr.plot(xs_fit, slope * xs_fit + intercept,
                         "k--", lw=1.3, alpha=0.6,
                         label=f"slope = {slope:.3f}")
            r_val = np.corrcoef(pv_v, ff_v)[0, 1]
            ax_corr.text(0.05, 0.07, f"Pearson r = {r_val:.3f}",
                         transform=ax_corr.transAxes, fontsize=9)
            ax_corr.legend(fontsize=8)
        ax_corr.set_xlabel(f"{param_choice}  ({param_label})", fontsize=10)
        ax_corr.set_ylabel("Final mean fitness")
        ax_corr.set_title("Fitness vs parameter (final generation)", fontsize=10)
        ax_corr.set_ylim(0, 1)
        ax_corr.grid(True, alpha=0.3)

        # Panel D: final phenotypic variance
        ax_var.scatter(pv_v, fv_v, c=cv_v, s=110, edgecolors="k",
                       linewidths=0.8, zorder=3)
        ax_var.set_xlabel(f"{param_choice}  ({param_label})", fontsize=10)
        ax_var.set_ylabel("Final phenotypic variance")
        ax_var.set_title(
            "Genetic diversity vs parameter (final generation)", fontsize=10
        )
        ax_var.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

        st.caption(
            "**Phase portrait**: trajectories in fitness–distance space. "
            "A well-adapted population drifts rightward then stabilises near the optimum. "
            "**Adaptation time**: first generation with mean fitness ≥ threshold; "
            "'never' bars reach slightly above the dashed max-generations line. "
            "**Correlation**: Pearson r quantifies the monotone relationship between "
            "the swept parameter and final fitness. "
            "**Diversity**: phenotypic variance proxies for standing genetic variation — "
            "a sharp drop signals a bottleneck or near-extinction event."
        )
