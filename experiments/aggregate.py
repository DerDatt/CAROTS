"""Aggregate CAROTS Step 1 results and render Step 2 localization figures.

After the runs finish (on Colab) and the ``results/`` folder is copied back, this
script:

1. Parses every ``results/<scenario>/seed<seed>/<DATA.NAME>/test.txt`` into a
   tidy table (scenario, anomaly_type, seed, AUROC, AUPRC, F1, ...).
2. Writes that table to ``results/summary.csv``.
3. Plots a baseline-vs-flawed AUROC comparison per anomaly type.
4. For each run that saved per-variable artifacts, calls
   ``experiments.localization.run`` to produce the Step 2 figures.

It depends only on numpy / pandas / matplotlib, so it runs locally without a GPU.
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
from experiments.scenarios import ANOMALIES, SCENARIOS  # noqa: E402

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


def plot_metric_comparison(df, metric="AUROC", out_path="results/comparison_auroc.png"):
    """Grouped bar chart: each anomaly type, one bar per scenario (mean over seeds)."""
    if not _HAS_MPL or df.empty:
        return None
    pivot = (df.groupby(["anomaly_type", "scenario"])[metric]
             .mean().unstack("scenario"))
    # Order scenarios with baseline first for readability.
    ordered = [s for s in SCENARIOS if s in pivot.columns]
    pivot = pivot[ordered]

    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel(metric)
    ax.set_xlabel("anomaly type")
    ax.set_title(f"CAROTS {metric}: baseline vs flawed training data")
    ax.legend(title="scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.0 if metric in ("AUROC", "AUPRC") else None)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_localization(df, base_dir="data/", out_root="results/localization", alpha=0.5):
    """Run the Step 2 localization for every result that saved per-variable data."""
    made = []
    for _, row in df.iterrows():
        result_dir = row["result_dir"]
        if not os.path.exists(os.path.join(result_dir, "per_variable_cd_error.npy")):
            continue
        data_dir = os.path.join(base_dir, SCENARIOS[row["scenario"]])
        out_dir = os.path.join(out_root, row["scenario"], f"seed{row['seed']}")
        try:
            L.run(result_dir=result_dir, data_dir=data_dir,
                  anomaly_type=row["anomaly_type"], factor=row["factor"],
                  win_size=VAR_WIN_SIZE, input_step=VAR_INPUT_STEP,
                  out_dir=out_dir, alpha=alpha)
            made.append(out_dir)
        except FileNotFoundError as e:
            print(f"  skipping localization for {result_dir}: {e}")
    return made


def main():
    parser = argparse.ArgumentParser(description="Aggregate CAROTS results + localization.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--base-dir", default="data/",
                        help="Where the data/VAR_<scenario> folders live.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Causal-propagation weight for localization.")
    parser.add_argument("--no-localization", action="store_true")
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

    for metric in ("AUROC", "AUPRC", "F1"):
        if metric in df.columns:
            p = plot_metric_comparison(
                df, metric, os.path.join(args.results_root, f"comparison_{metric.lower()}.png"))
            if p:
                print(f"wrote {p}")

    if not args.no_localization:
        made = render_localization(df, base_dir=args.base_dir,
                                   out_root=os.path.join(args.results_root, "localization"),
                                   alpha=args.alpha)
        print(f"wrote localization figures for {len(made)} runs")


if __name__ == "__main__":
    main()
