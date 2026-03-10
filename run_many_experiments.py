#!/usr/bin/env python3
"""
run_many_experiments.py — Batch runner for multiple GFM experiment configs.

Resolves a list of targets (directories, glob patterns, or individual files)
into a sorted, deduplicated list of JSON configs, then runs each one in
sequence using the same logic as run_experiment.py.

Usage examples
--------------
# All configs in a subfolder:
    python run_many_experiments.py experiments/drift_exploration/

# Shell-glob pattern (quote to prevent shell expansion if needed):
    python run_many_experiments.py "experiments/drift_c*"

# Named prefix inside the default experiments/ directory:
    python run_many_experiments.py --prefix drift_c
    python run_many_experiments.py --prefix pop_N --prefix mut_xi

# Mix of the above:
    python run_many_experiments.py experiments/baseline.json "experiments/sel_*"

# Skip the confirmation prompt:
    python run_many_experiments.py experiments/drift_exploration/ --yes

# Limit parallel workers per experiment:
    python run_many_experiments.py "experiments/drift_c*" --workers 4

# Dry-run (list configs without running):
    python run_many_experiments.py "experiments/drift_c*" --dry-run
"""

import argparse
import glob
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_targets(targets: list[str], prefixes: list[str]) -> list[Path]:
    """
    Turn a mix of directories, glob patterns and file paths into a
    deduplicated, sorted list of .json config files.
    """
    found: list[Path] = []

    # 1. Named prefixes → search experiments/ directory
    for prefix in prefixes:
        matched = sorted(EXPERIMENTS_DIR.glob(f"{prefix}*.json"))
        if not matched:
            print(f"  [warn] --prefix '{prefix}' matched no files in {EXPERIMENTS_DIR}/")
        found.extend(matched)

    # 2. Positional targets
    for target in targets:
        p = Path(target)

        if p.is_dir():
            # Entire directory — all .json files (non-recursive by default)
            dir_files = sorted(p.glob("*.json"))
            if not dir_files:
                print(f"  [warn] directory '{target}' contains no .json files")
            found.extend(dir_files)

        elif "*" in target or "?" in target or "[" in target:
            # Glob pattern (handles both shell-expanded and quoted globs)
            matched = sorted(Path(g) for g in glob.glob(target))
            matched = [m for m in matched if m.suffix == ".json"]
            if not matched:
                print(f"  [warn] glob '{target}' matched no .json files")
            found.extend(matched)

        elif p.exists():
            if p.suffix != ".json":
                print(f"  [warn] '{target}' is not a .json file — skipping")
            else:
                found.append(p)

        else:
            # Maybe the user typed a bare prefix without --prefix
            by_prefix = sorted(EXPERIMENTS_DIR.glob(f"{target}*.json"))
            if by_prefix:
                print(f"  [info] '{target}' not found as path; "
                      f"interpreted as prefix → {len(by_prefix)} file(s)")
                found.extend(by_prefix)
            else:
                print(f"  [warn] '{target}' not found as path or prefix — skipping")

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple GFM experiment configs in sequence.\n\n"
            "Targets can be directories, glob patterns, or individual .json files.\n"
            "Each config is run with run_experiment.run_one()."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_many_experiments.py experiments/drift_exploration/\n"
            "  python run_many_experiments.py \"experiments/drift_c*\"\n"
            "  python run_many_experiments.py --prefix drift_c --prefix pop_N\n"
            "  python run_many_experiments.py \"experiments/sel_*\" --workers 4 --yes\n"
        ),
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="TARGET",
        help=(
            "One or more of: a directory path, a glob pattern "
            "(quoted), or a specific .json file."
        ),
    )
    parser.add_argument(
        "--prefix", "-p",
        action="append",
        dest="prefixes",
        default=[],
        metavar="PREFIX",
        help=(
            "Match all experiments/<PREFIX>*.json files. "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help="Parallel worker processes per experiment (default: n_replicates).",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt and run immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config list and exit without running anything.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch if any single experiment fails (default: continue).",
    )

    args = parser.parse_args()

    if not args.targets and not args.prefixes:
        parser.error("Provide at least one TARGET or --prefix.")

    # ---- Resolve configs ----
    configs = _resolve_targets(args.targets, args.prefixes)

    if not configs:
        sys.exit("No valid .json configs found. Nothing to run.")

    # ---- Preview ----
    total_reps = 0
    import json
    rows = []
    for cfg_path in configs:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                meta = json.load(f)
            n_r = meta.get("n_replicates", "?")
            name = meta.get("name", cfg_path.stem)
            desc = meta.get("description", "")
        except Exception:
            n_r, name, desc = "?", cfg_path.stem, "(could not parse)"
        rows.append((cfg_path, name, n_r, desc))
        if isinstance(n_r, int):
            total_reps += n_r

    print(f"\n{'─'*62}")
    print(f"  Configs to run : {len(configs)}")
    print(f"  Total replicates: {total_reps}")
    print(f"{'─'*62}")
    for i, (cfg_path, name, n_r, desc) in enumerate(rows, 1):
        desc_str = f"  {desc}" if desc else ""
        print(f"  {i:2d}. {name:<30s}  ({n_r} reps)  {cfg_path}{desc_str}")
    print(f"{'─'*62}\n")

    if args.dry_run:
        print("Dry-run mode — exiting without running.")
        return

    if not args.yes:
        try:
            answer = input(f"Run all {len(configs)} experiment(s)? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    # ---- Import after confirmation so startup is fast ----
    from run_experiment import run_one

    # ---- Run sequentially ----
    succeeded: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for i, (cfg_path, name, n_r, _) in enumerate(rows, 1):
        print(f"\n{'═'*62}")
        print(f"  [{i}/{len(configs)}]  {name}")
        print(f"{'═'*62}")
        try:
            out_dir = run_one(cfg_path, n_workers=args.workers)
            succeeded.append(out_dir)
        except SystemExit as exc:
            msg = str(exc)
            print(f"  ✗ FAILED: {msg}")
            failed.append((cfg_path, msg))
            if args.stop_on_error:
                sys.exit(f"Stopping batch after failure in '{name}'.")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ FAILED: {msg}")
            failed.append((cfg_path, msg))
            if args.stop_on_error:
                sys.exit(f"Stopping batch after failure in '{name}'.")

    # ---- Final summary ----
    print(f"\n{'═'*62}")
    print(f"  Batch complete")
    print(f"  ✓ Succeeded : {len(succeeded)}")
    if failed:
        print(f"  ✗ Failed    : {len(failed)}")
        for cfg_path, msg in failed:
            print(f"      {cfg_path}: {msg}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
