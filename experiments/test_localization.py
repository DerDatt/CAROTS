"""Self-contained smoke test for the Step 2 offline analysis (CPU, no GPU).

Why this exists
---------------
The localization metrics are computed after a cluster/Colab run that takes
hours. A crash in the *offline* analysis is therefore expensive: it is only
discovered once the GPU time is already spent. This test fabricates the same
``.npy`` artifacts a real run would write, on a tiny synthetic causal chain with
a *known* culprit, and checks that:

1. ``localization.run`` works end to end and writes its figures and metrics,
2. a planted anomaly on the root variable is actually recovered
   (``hit@1`` well above chance, ``var_auroc`` well above 0.5),
3. the two-sided score recovers a culprit whose forecasting error *drops*
   (the collective-global failure mode), where the signed score fails,
4. ``aggregate`` walks a results tree, builds the summary tables and computes
   the seed spread without touching a GPU.

Run it before shipping a run to the cluster::

    python -m experiments.test_localization
"""

import os
import shutil
import tempfile

import numpy as np

from experiments import aggregate as A
from experiments import localization as L

N_VARS = 5
N_WINDOWS = 400
WIN_SIZE = 4
INPUT_STEP = 3
ROOT_VAR = 0


def _chain_graph(n_vars=N_VARS, chain_len=3):
    """GC[i, j] = 1 means i causes j; a chain 0 -> 1 -> ... plus self-loops."""
    graph = np.eye(n_vars, dtype=int)
    for k in range(chain_len - 1):
        graph[k, k + 1] = 1
    return graph


def _fabricate(tmp, anomaly_type, factor, error_direction="up", seed=0):
    """Write the artifacts a real CAROTS run would produce.

    Args:
        error_direction: ``"up"`` plants a forecasting-error *spike* on the root
            (the point/trend case); ``"down"`` plants a *drop* (the
            collective-global case, where the injected segment is smoother than
            the real signal and therefore easier to forecast).
    """
    rng = np.random.RandomState(seed)
    result_dir = os.path.join(tmp, "results", "toy", "seed0", f"VAR_{anomaly_type}")
    data_dir = os.path.join(tmp, "data", "VAR_toy")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Baseline forecasting error: strictly positive, variable-specific scale.
    scale = np.array([1.0, 2.0, 0.5, 1.5, 0.8])
    error = np.abs(rng.normal(size=(N_WINDOWS, N_VARS))) * scale + 0.5

    # Ground truth lives on the timestep axis; window t covers the target
    # timesteps [t + INPUT_STEP, t + WIN_SIZE), so timestep t + INPUT_STEP is
    # the one that lands in window t.
    n_timesteps = N_WINDOWS + WIN_SIZE
    var_labels = np.zeros((n_timesteps, N_VARS), dtype=int)
    labels = np.zeros(n_timesteps, dtype=int)

    anomalous_windows = np.arange(20, N_WINDOWS, 40)
    for w in anomalous_windows:
        if error_direction == "up":
            error[w, ROOT_VAR] *= 50.0          # culprit becomes unforecastable
            error[w, ROOT_VAR + 1] *= 8.0       # leakage into the causal child
        else:
            error[w, ROOT_VAR] *= 0.01          # culprit becomes *too* easy
            error[w, ROOT_VAR + 1] *= 6.0
        var_labels[w + INPUT_STEP, ROOT_VAR] = 1
        labels[w + INPUT_STEP] = 1

    np.save(os.path.join(result_dir, "per_variable_cd_error.npy"), error)
    np.save(os.path.join(result_dir, "causality_matrix.npy"), _chain_graph())
    stem = f"test_{anomaly_type}_outliers_factor{factor}"
    np.save(os.path.join(data_dir, f"{stem}_varlabels.npy"), var_labels)
    np.save(os.path.join(data_dir, f"{stem}_labels.npy"), labels)
    return result_dir, data_dir


def _run_case(tmp, anomaly_type, factor, error_direction):
    result_dir, data_dir = _fabricate(tmp, anomaly_type, factor, error_direction)
    out_dir = os.path.join(tmp, "figures", anomaly_type)
    return L.run(result_dir=result_dir, data_dir=data_dir,
                 anomaly_type=anomaly_type, factor=factor,
                 win_size=WIN_SIZE, input_step=INPUT_STEP, out_dir=out_dir), out_dir


def _fabricate_results_tree(root, scenarios=("baseline", "toy_chain",
                                             "toy_chain_mid"),
                            seeds=(2, 3)):
    """Build a miniature ``results/`` + ``data/`` tree as a real grid would.

    Uses registered scenario names so ``aggregate`` can map each run back to its
    dataset folder via ``scenarios.SCENARIOS``.
    """
    for scenario in scenarios:
        data_dir = os.path.join(root, "data", f"VAR_{scenario}")
        for seed in seeds:
            for anomaly, factor in (("point_global", "3.0"),
                                    ("collective_global", "None")):
                result_dir = os.path.join(root, "results", scenario, f"seed{seed}",
                                          f"VAR_{anomaly}_factor{factor}")
                os.makedirs(result_dir, exist_ok=True)
                os.makedirs(data_dir, exist_ok=True)

                rng = np.random.RandomState(seed)
                error = np.abs(rng.normal(size=(N_WINDOWS, N_VARS))) + 0.5
                n_timesteps = N_WINDOWS + WIN_SIZE
                var_labels = np.zeros((n_timesteps, N_VARS), dtype=int)
                labels = np.zeros(n_timesteps, dtype=int)
                for w in range(20, N_WINDOWS, 40):
                    error[w, ROOT_VAR] *= 30.0
                    var_labels[w + INPUT_STEP, ROOT_VAR] = 1
                    labels[w + INPUT_STEP] = 1

                np.save(os.path.join(result_dir, "per_variable_cd_error.npy"), error)
                np.save(os.path.join(result_dir, "causality_matrix.npy"),
                        _chain_graph())
                stem = f"test_{anomaly}_outliers_factor{factor}"
                np.save(os.path.join(data_dir, f"{stem}_varlabels.npy"), var_labels)
                np.save(os.path.join(data_dir, f"{stem}_labels.npy"), labels)
                with open(os.path.join(result_dir, "test.txt"), "w") as f:
                    f.write(f"AUROC: {0.7 + 0.01 * seed:.3f}, "
                            f"AUPRC: {0.3 + 0.01 * seed:.3f}, "
                            f"F1: {0.5 + 0.01 * seed:.3f}\n")


def _check_aggregate(tmp):
    """Exercise the offline aggregation exactly as it runs after a real grid."""
    root = os.path.join(tmp, "agg")
    _fabricate_results_tree(root)
    results_root = os.path.join(root, "results")

    df = A.collect_results(results_root)
    assert len(df) == 12, f"expected 3 scenarios x 2 seeds x 2 anomalies, got {len(df)}"
    assert set(df["scenario"]) == {"baseline", "toy_chain", "toy_chain_mid"}, \
        df["scenario"].unique()

    seeds = A.summarize_over_seeds(df)
    assert not seeds.empty and "AUROC_std" in seeds.columns, seeds.columns.tolist()
    assert (seeds["AUROC_count"] == 2).all(), seeds

    plot = A.plot_metric_comparison(
        df, "AUROC", os.path.join(results_root, "comparison_auroc.png"))
    assert plot and os.path.exists(plot), "comparison plot with error bars missing"
    # The toy scenarios have a different dimensionality and anomaly factor, so
    # the Step-1 chart must leave them out; a toy-only tree yields no chart.
    toys_only = df[df["scenario"] != "baseline"]
    assert A.plot_metric_comparison(
        toys_only, "AUROC", os.path.join(results_root, "toys_only.png")) is None

    metrics_df, sweep_df = A.render_localization(
        df, base_dir=os.path.join(root, "data"),
        out_root=os.path.join(results_root, "localization"))
    # 12 runs x {direct, causal} x {signed, two_sided}
    assert len(metrics_df) == 48, len(metrics_df)
    assert set(metrics_df["score_mode"]) == {"signed", "two_sided"}, metrics_df
    assert {"auroc_within", "var_auroc", "mrr", "hit@1", "leak_children"} \
        <= set(metrics_df.columns)
    assert len(sweep_df) == 12 * 2 * len(L.DEFAULT_ALPHAS), len(sweep_df)
    return df, seeds, metrics_df


def main():
    tmp = tempfile.mkdtemp(prefix="carots_loc_test_")
    try:
        print("=== case 1: error spike on the root (point/trend-like) ===")
        summary, out_dir = _run_case(tmp, "point_global", "3.0", "up")
        signed = summary["direct_signed"]
        assert summary["direct_signed"]["n_windows"] == 10, summary
        assert signed["hit@1"] == 1.0, f"expected perfect hit@1, got {signed}"
        assert signed["var_auroc"] > 0.9, signed
        assert signed["hit@1"] > signed["random_hit@1"], signed
        print("  OK: signed score localizes the culprit\n")

        print("=== case 2: error DROP on the root (collective-global-like) ===")
        # The signed score ranks a culprit whose error *drops* as maximally
        # normal, i.e. dead last. The two-sided score repairs that, but only
        # partly: a drop is bounded below by zero error, so its |z| stays small,
        # while a spiking causal child is unbounded and can still outrank the
        # true culprit. Causal propagation is what closes the remaining gap.
        summary, _ = _run_case(tmp, "collective_global", "None", "down")
        signed = summary["direct_signed"]
        two_sided = summary["direct_two_sided"]
        causal_two_sided = summary["causal_two_sided"]
        assert signed["var_auroc"] < 0.5, (
            f"signed score should rank the culprit below chance, got {signed}")
        assert two_sided["var_auroc"] > 0.8, (
            f"two-sided score should recover the culprit, got {two_sided}")
        assert two_sided["hit@1"] > two_sided["random_hit@1"], two_sided
        assert causal_two_sided["var_auroc"] >= two_sided["var_auroc"], (
            "propagation should not hurt when the child carries the evidence: "
            f"{causal_two_sided} vs {two_sided}")
        for name in ("direct_signed", "direct_two_sided", "causal_two_sided"):
            m = summary[name]
            print(f"  {name:<18} var_auroc={m['var_auroc']:.3f}  "
                  f"hit@1={m['hit@1']:.2f}  mrr={m['mrr']:.2f}")
        print("  OK: two-sided score fixes the sign problem\n")

        print("=== case 3: alpha sweep + leakage diagnostic ===")
        sweep = summary["alpha_sweep"]
        assert {r["score_mode"] for r in sweep} == {"signed", "two_sided"}, sweep
        assert any(r["alpha"] == 0.0 for r in sweep), sweep
        leak = summary["leakage"]
        assert leak["children"] > leak["other"], (
            f"planted leakage should lift the children, got {leak}")
        print(f"  leakage: culprit={leak['culprit']:.2f} "
              f"children={leak['children']:.2f} other={leak['other']:.2f}")
        print(f"  sweep rows: {len(sweep)}")

        figures = sorted(os.listdir(out_dir))
        assert any(f.endswith("_alpha_sweep.png") for f in figures), figures
        assert any("two_sided_heatmap" in f for f in figures), figures
        print(f"  wrote {len(figures)} figures, e.g. {figures[0]}\n")

        print("=== case 4: offline aggregation over a full results tree ===")
        df, seeds, metrics_df = _check_aggregate(tmp)
        print(f"  summary.csv rows            : {len(df)}")
        print(f"  summary_by_seed.csv rows    : {len(seeds)} (with std columns)")
        print(f"  localization_metrics rows   : {len(metrics_df)}")
        print("  OK: aggregation runs without a GPU")

        print("\nAll localization smoke tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
