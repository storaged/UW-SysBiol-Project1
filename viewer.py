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
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GFM Experiment Viewer",
    page_icon="🧬",
    layout="wide",
)

RESULTS_DIR = Path("results")

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def discover_runs(results_dir: Path) -> list[dict]:
    """
    Scan results/ for subdirectories that contain config.json + summary.csv.
    Returns a list of dicts, one per run, sorted newest-first.
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
        runs.append({
            "dir":       d,
            "label":     d.name,                          # <name>_<timestamp>
            "name":      cfg.get("name", d.name),
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

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🧬 GFM Viewer")
st.sidebar.markdown("---")

runs = discover_runs(RESULTS_DIR)

if not runs:
    st.title("No experiment results found")
    st.markdown(
        f"Run an experiment first:\n```bash\n"
        f"python run_experiment.py experiments/baseline.json --workers 5\n```\n"
        f"Results will appear in **`{RESULTS_DIR}/`**."
    )
    st.stop()

run_labels = [r["label"] for r in runs]
run_by_label = {r["label"]: r for r in runs}

page = st.sidebar.radio("Page", ["Overview", "Single run", "Compare two runs"])

# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Experiment overview")
    rows = []
    for r in runs:
        summary = load_summary(r["dir"])
        extinct = int(summary["extinct_count"].max()) if "extinct_count" in summary.columns else "?"
        last_fit_mean = summary["mean_fitness_mean"].iloc[-1] if "mean_fitness_mean" in summary.columns else float("nan")
        rows.append({
            "Run": r["label"],
            "Name": r["name"],
            "Replicates": r["n_reps"],
            "n": r["cfg"].get("n", "?"),
            "N": r["cfg"].get("N", "?"),
            "c": r["cfg"].get("c", "?"),
            "sigma": r["cfg"].get("sigma", "?"),
            "max_gen": r["cfg"].get("max_generations", "?"),
            "Extinct": extinct,
            "Final mean fitness": f"{last_fit_mean:.4f}" if isinstance(last_fit_mean, float) else "?",
            "Git commit": r["git"],
            "Timestamp": r["timestamp"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        f"Showing {len(runs)} run(s) from `{RESULTS_DIR}/`. "
        "Add more runs with `python run_experiment.py experiments/<config>.json`."
    )

# ── Page: Single run ─────────────────────────────────────────────────────────

elif page == "Single run":
    st.title("Single run detail")

    sel = st.sidebar.selectbox("Select run", run_labels)
    run = run_by_label[sel]

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

    sel_a = st.sidebar.selectbox("Condition A", run_labels, index=0)
    sel_b = st.sidebar.selectbox(
        "Condition B",
        run_labels,
        index=min(1, len(run_labels) - 1),
    )

    run_a = run_by_label[sel_a]
    run_b = run_by_label[sel_b]

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
