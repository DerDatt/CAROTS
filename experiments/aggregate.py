"""Aggregate CAROTS Step 1 results and render Step 2 localization figures.

After the runs finish (on Colab) and the ``results/`` folder is copied back, this
script:

1. Parses every ``results/<scenario>/seed<seed>/<DATA.NAME>/test.txt`` into a
   tidy table (scenario, anomaly_type, seed, AUROC, AUPRC, F1, ...).
2. Writes that table to ``results/summary.csv`` and its mean/std over seeds to
   ``results/summary_by_seed.csv``.
3. Plots a baseline-vs-flawed comparison per anomaly type, with error bars once
   more than one seed is available.
4. For each run that saved per-variable artifacts, calls
   ``experiments.localization.run`` to produce the Step 2 figures **and** the
   quantitative localization metrics, written to
   ``results/localization_metrics.csv`` and
   ``results/localization_alpha_sweep.csv``.

It depends only on numpy / pandas / matplotlib / scikit-learn, so it runs
locally without a GPU.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import localization as L  # noqa: E402
from experiments.scenarios import ANOMALIES, SCENARIOS, STEP1_SCENARIOS  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

# For VAR, utils/parser.py forces these inference settings; the localization
# alignment must use the same numbers.
VAR_WIN_SIZE = 4
VAR_INPUT_STEP = 3

_METRIC_RE = re.compile(r"([A-Za-z0-9_]+):\s*([-+0-9.eE]+)")


def parse_test_txt(path):
    """Parse a CAROTS ``test.txt`` ('AUROC: 0.99, AUPRC: 0.88, ...') into a dict."""
    with open(path) as f:
        text = f.read()
    return {k: float(v) for k, v in _METRIC_RE.findall(text)}


def _name_to_anomaly(dataset_name):
    """Recover (anomaly_type, factor) from a DATA.NAME like VAR_point_global_factor2.0."""
    for spec in ANOMALIES:
        if dataset_name == spec.dataset_name:
            return spec.name, spec.factor
    # Fall back to a generic parse.
    body = dataset_name[len("VAR_"):]
    anomaly, factor = body.rsplit("_factor", 1)
    return anomaly, factor


def collect_results(results_root="results"):
    """Walk the results tree and return a tidy DataFrame of all metrics."""
    rows = []
    pattern = os.path.join(results_root, "*", "seed*", "VAR_*", "test.txt")
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        # .../<scenario>/seed<seed>/<DATA.NAME>/test.txt
        scenario = parts[-4]
        seed = int(parts[-3].replace("seed", ""))
        dataset_name = parts[-2]
        anomaly, factor = _name_to_anomaly(dataset_name)
        metrics = parse_test_txt(path)
        rows.append({
            "scenario": scenario, "anomaly_type": anomaly, "factor": factor,
            "seed": seed, **metrics, "result_dir": os.path.dirname(path),
        })
    return pd.DataFrame(rows)


def summarize_over_seeds(df, metrics=("AUROC", "AUPRC", "F1")):
    """Collapse the per-seed table into mean / std / count per (scenario, anomaly).

    With a single seed the std column is NaN, which is itself the honest answer:
    no spread can be estimated. Run more seeds to get error bars.
    """
    present = [m for m in metrics if m in df.columns]
    if df.empty or not present:
        return pd.DataFrame()
    grouped = df.groupby(["scenario", "anomaly_type"])[present]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


def plot_metric_comparison(df, metric="AUROC", out_path="results/comparison_auroc.png"):
    """Grouped bar chart: each anomaly type, one bar per scenario.

    Bars are the mean over seeds; when more than one seed is available the
    standard deviation is drawn as an error bar, so it is visible whether a
    scenario difference exceeds the seed-to-seed noise.
    """
    if not _HAS_MPL or df.empty:
        return None
    # The toy scenarios use a different dimensionality and anomaly factor, so
    # putting them next to the Step-1 bars would invite a comparison that is not
    # meaningful.
    df = df[df["scenario"].isin(STEP1_SCENARIOS)]
    if df.empty:
        return None

    grouped = df.groupby(["anomaly_type", "scenario"])[metric]
    pivot = grouped.mean().unstack("scenario")
    spread = grouped.std().unstack("scenario")
    seeds_per_scenario = df.groupby("scenario")["seed"].nunique()
    max_seeds = int(seeds_per_scenario.max())
    # Order scenarios with baseline first for readability.
    ordered = [s for s in STEP1_SCENARIOS if s in pivot.columns]
    pivot = pivot[ordered]
    spread = spread.reindex(columns=ordered)

    ax = pivot.plot(kind="bar", figsize=(11, 5),
                    yerr=spread if max_seeds > 1 else None, capsize=3)
    ax.set_ylabel(metric)
    ax.set_xlabel("anomaly type")
    if max_seeds <= 1:
        seed_note = "single seed - no error bars"
    else:
        lo, hi = int(seeds_per_scenario.min()), max_seeds
        span = f"{hi}" if lo == hi else f"{lo}-{hi}"
        seed_note = f"mean +/- std over {span} seeds per scenario"
    ax.set_title(f"CAROTS {metric}: baseline vs flawed training data ({seed_note})")
    ax.legend(title="scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.0 if metric in ("AUROC", "AUPRC") else None)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_localization(df, base_dir="data/", out_root="results/localization", alpha=0.5):
    """Run the Step 2 localization for every result that saved per-variable data.

    Besides the figures (written by ``localization.run``), this collects the
    quantitative variable-level metrics, so Step 2 can be reported as numbers
    rather than eyeballed heatmaps.

    Returns:
        (metrics_df, sweep_df): one row per (run, attribution) with the
        variable-level metrics and the leakage diagnostic, and one row per
        (run, alpha) with the propagation sweep. Both are empty when no run
        saved per-variable artifacts.
    """
    metric_rows, sweep_rows = [], []
    for _, row in df.iterrows():
        result_dir = row["result_dir"]
        if not os.path.exists(os.path.join(result_dir, "per_variable_cd_error.npy")):
            continue
        if row["scenario"] not in SCENARIOS:
            # A results folder that is not part of the registered grid (e.g. an
            # archived run). Skip it instead of crashing the whole aggregation.
            print(f"  skipping localization for unknown scenario "
                  f"{row['scenario']!r}")
            continue
        data_dir = os.path.join(base_dir, SCENARIOS[row["scenario"]])
        out_dir = os.path.join(out_root, row["scenario"], f"seed{row['seed']}")
        key = {"scenario": row["scenario"], "anomaly_type": row["anomaly_type"],
               "seed": row["seed"]}
        try:
            summary = L.run(result_dir=result_dir, data_dir=data_dir,
                            anomaly_type=row["anomaly_type"], factor=row["factor"],
                            win_size=VAR_WIN_SIZE, input_step=VAR_INPUT_STEP,
                            out_dir=out_dir, alpha=alpha)
        except FileNotFoundError as e:
            print(f"  skipping localization for {result_dir}: {e}")
            continue

        # The leakage diagnostic is a property of the run, not of a single
        # attribution, so it is repeated on every row for easy filtering.
        leakage = {f"leak_{k}": v for k, v in summary["leakage"].items()}
        for name, metrics in summary.items():
            if name in ("alpha_sweep", "leakage"):
                continue
            attribution, _, score_mode = name.partition("_")
            metric_rows.append({**key, "attribution": attribution,
                                "score_mode": score_mode, **metrics, **leakage})
        for sweep in summary["alpha_sweep"]:
            sweep_rows.append({**key, **sweep})

    return pd.DataFrame(metric_rows), pd.DataFrame(sweep_rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CAROTS results + localization.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--base-dir", default="data/",
                        help="Where the data/VAR_<scenario> folders live.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Causal-propagation weight for localization.")
    parser.add_argument("--no-localization", action="store_true")
    parser.add_argument("--localization-scenarios", nargs="*", default=None,
                        help="Restrict the Step 2 localization to these "
                             "scenarios. Useful because the per-variable "
                             "evaluation over a 128-variable grid with many "
                             "seeds is far slower than the metric parsing.")
    parser.add_argument("--localization-seeds", nargs="*", type=int, default=None,
                        help="Restrict the Step 2 localization to these seeds.")
    args = parser.parse_args()

    df = collect_results(args.results_root)
    if df.empty:
        print(f"No results found under {args.results_root}/ "
              f"(expected <scenario>/seed*/VAR_*/test.txt).")
        return

    csv_path = os.path.join(args.results_root, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")
    print(df.drop(columns=["result_dir"]).to_string(index=False))

    seeds = summarize_over_seeds(df)
    if not seeds.empty:
        seeds_path = os.path.join(args.results_root, "summary_by_seed.csv")
        seeds.to_csv(seeds_path, index=False)
        print(f"\nwrote {seeds_path} (mean/std over seeds)")
        print(seeds.to_string(index=False))

    for metric in ("AUROC", "AUPRC", "F1"):
        if metric in df.columns:
            p = plot_metric_comparison(
                df, metric, os.path.join(args.results_root, f"comparison_{metric.lower()}.png"))
            if p:
                print(f"wrote {p}")

    if not args.no_localization:
        subset = df
        if args.localization_scenarios:
            subset = subset[subset["scenario"].isin(args.localization_scenarios)]
        if args.localization_seeds:
            subset = subset[subset["seed"].isin(args.localization_seeds)]
        metrics_df, sweep_df = render_localization(
            subset, base_dir=args.base_dir,
            out_root=os.path.join(args.results_root, "localization"),
            alpha=args.alpha)
        if metrics_df.empty:
            print("no run saved per-variable artifacts "
                  "(re-run with TEST.SAVE_PER_VARIABLE True) - skipping Step 2")
        else:
            m_path = os.path.join(args.results_root, "localization_metrics.csv")
            metrics_df.to_csv(m_path, index=False)
            print(f"\nwrote {m_path} ({len(metrics_df)} rows)")
            show = [c for c in ("scenario", "anomaly_type", "seed", "attribution",
                                "score_mode", "auroc_within", "var_auroc",
                                "var_ap", "random_ap", "mrr", "hit@1",
                                "random_hit@1", "n_windows")
                    if c in metrics_df.columns]
            print(metrics_df[show].to_string(index=False))

            s_path = os.path.join(args.results_root, "localization_alpha_sweep.csv")
            sweep_df.to_csv(s_path, index=False)
            print(f"wrote {s_path} ({len(sweep_df)} rows)")


if __name__ == "__main__":
    main()
