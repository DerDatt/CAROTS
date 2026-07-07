"""Step 2 - Variable-level anomaly localization for CAROTS (inference-only).

CAROTS tells you *when* a multivariate time series is anomalous (one anomaly
score per window). This module answers the follow-up question *which variable*
is responsible, **without changing the training procedure, the model or the loss
functions**. It is a pure post-processing layer on top of a trained model.

How it works
------------
The causal discoverer inside CAROTS forecasts every variable from its causal
parents. During inference we keep the squared forecasting error *per variable*
instead of averaging it away (see ``Scorer.get_per_variable_cd_scores`` and the
``TEST.SAVE_PER_VARIABLE`` hook in the Predictor). That gives a matrix

    E[t, n] = forecasting error of variable ``n`` in test window ``t``.

A variable that suddenly becomes hard to forecast is a good anomaly candidate.
But forecasting error also *leaks downstream*: if a root-cause variable is
corrupted, its causal children inherit a corrupted input and look anomalous too.
We therefore offer two attributions:

1. ``direct``  - use the per-variable error as-is (optionally robust-normalized
   so every variable is judged against its own typical error level).
2. ``causal``  - propagate evidence along the *known* causal graph so that a
   variable is boosted when its causal children are also anomalous. This nudges
   the attribution toward upstream root causes.

This file is meant to run **offline / on CPU**: it only consumes the ``.npy``
artifacts produced during inference, so you can iterate on the visualizations
locally without a GPU.

Inputs it expects (produced by a CAROTS run with ``TEST.SAVE_PER_VARIABLE=True``)
    <result_dir>/per_variable_cd_error.npy   shape (n_windows, N)
    <result_dir>/causality_matrix.npy        shape (N, N), A[i, j]=1 => i causes j
And from the dataset folder (produced by experiments/datagen.py)
    <data_dir>/test_<type>_outliers_factor<f>_varlabels.npy   shape (T, N)
    <data_dir>/test_<type>_outliers_factor<f>_labels.npy      shape (T,)
"""

import argparse
import os

import numpy as np

# Matplotlib is only needed for the plotting helpers. Import lazily so the
# numeric functions still work in environments without a display backend.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend; good for Colab/servers/CI.
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_localization_artifacts(result_dir, data_dir, anomaly_type, factor):
    """Load everything the localization needs from disk.

    Args:
        result_dir: Folder with the CAROTS run outputs (per_variable_cd_error.npy
            and causality_matrix.npy).
        data_dir: Dataset folder produced by experiments/datagen.py.
        anomaly_type: e.g. ``"point_global"`` (matches the test file name).
        factor: The factor string/number used in the test file name, e.g.
            ``2.0`` or ``"None"`` for collective-global.

    Returns:
        dict with keys ``error`` (n_windows, N), ``graph`` (N, N),
        ``var_labels`` (T, N) and ``labels`` (T,).
    """
    error = np.load(os.path.join(result_dir, "per_variable_cd_error.npy"))
    graph = np.load(os.path.join(result_dir, "causality_matrix.npy"))

    stem = f"test_{anomaly_type}_outliers_factor{factor}"
    var_labels = np.load(os.path.join(data_dir, f"{stem}_varlabels.npy"))
    labels = np.load(os.path.join(data_dir, f"{stem}_labels.npy"))

    return {"error": error, "graph": graph, "var_labels": var_labels, "labels": labels}


def window_var_labels(var_labels, win_size, input_step, step=1, n_windows=None):
    """Align per-timestep variable labels to the per-window error matrix.

    The causal discoverer forecasts the *target* part of each window, i.e. the
    timesteps ``[input_step:win_size]``. Window ``t`` starts at test timestep
    ``t*step``, so its target timesteps are ``t*step + input_step ...
    t*step + win_size - 1``. A variable is "truly anomalous in window ``t``" if
    any of those target timesteps is labelled anomalous for that variable.

    Args:
        var_labels: Per-timestep, per-variable ground truth, shape (T, N).
        win_size: Window length used at inference.
        input_step: Number of leading timesteps fed to the causal discoverer.
        step: Window stride (CAROTS uses 1 at test time).
        n_windows: Optional expected number of windows (to clip rounding diffs).

    Returns:
        Boolean array of shape (n_windows, N): the per-window variable ground
        truth, aligned with ``per_variable_cd_error``.
    """
    T, N = var_labels.shape
    total = (T - win_size) // step + 1
    masks = np.zeros((total, N), dtype=bool)
    for t in range(total):
        start = t * step + input_step
        end = t * step + win_size
        masks[t] = var_labels[start:end].sum(axis=0) > 0
    if n_windows is not None and n_windows != total:
        # Clip/pad so the mask lines up exactly with the saved error matrix.
        masks = masks[:n_windows]
        if masks.shape[0] < n_windows:
            pad = np.zeros((n_windows - masks.shape[0], N), dtype=bool)
            masks = np.concatenate([masks, pad], axis=0)
    return masks


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def robust_zscore(error, eps=1e-8):
    """Normalize each variable's error by its own robust baseline.

    Most test windows are normal, so the per-variable median and MAD (median
    absolute deviation) estimate that variable's typical error. Dividing by them
    puts every variable on a comparable scale, so a spike stands out even for a
    variable whose raw error is naturally small.

    Args:
        error: Per-variable error matrix, shape (n_windows, N).

    Returns:
        Robustly standardized matrix of the same shape.
    """
    median = np.median(error, axis=0, keepdims=True)
    mad = np.median(np.abs(error - median), axis=0, keepdims=True)
    # 1.4826 makes MAD a consistent estimator of std for Gaussian data.
    return (error - median) / (1.4826 * mad + eps)


def localize_direct(error, normalize=True):
    """``direct`` attribution: the per-variable error itself.

    Args:
        error: Per-variable error matrix, shape (n_windows, N).
        normalize: If True, robust-zscore each variable first (recommended).

    Returns:
        Score matrix (n_windows, N); higher means "more likely the culprit".
    """
    return robust_zscore(error) if normalize else error


def _row_normalize(graph):
    """Row-normalize the adjacency so each row averages over a node's children."""
    out_degree = graph.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1.0  # avoid div-by-zero for childless nodes
    return graph / out_degree


def localize_causal(error, graph, alpha=0.5, normalize=True):
    """``causal`` attribution: propagate evidence to upstream root causes.

    For each variable we add a fraction ``alpha`` of the *mean error of its
    causal children*. Intuition: if variable ``i`` is the true root cause, then
    not only is ``i`` hard to forecast, but the children it feeds into are hard
    to forecast as well. Accumulating children's evidence at the parent makes
    the upstream source stand out from the downstream variables that merely
    inherited the disturbance.

    With ``A[i, j] = 1`` meaning "i causes j" and ``A_norm`` row-normalized:

        loc[t, i] = score[t, i] + alpha * mean_{j in children(i)} score[t, j]
                  = score + alpha * (score @ A_norm.T)

    Args:
        error: Per-variable error matrix, shape (n_windows, N).
        graph: Binarized causal graph, shape (N, N), ``A[i, j]=1`` => i causes j.
        alpha: Weight of the propagated child evidence (0 disables propagation,
            recovering ``direct``).
        normalize: If True, robust-zscore the error before propagating.

    Returns:
        Score matrix (n_windows, N).
    """
    score = robust_zscore(error) if normalize else error
    # Drop self-loops on the diagonal so a variable does not "propagate to
    # itself"; the diagonal of CAROTS' causality matrix is typically 1.
    children = graph.copy()
    np.fill_diagonal(children, 0.0)
    a_norm = _row_normalize(children)
    child_evidence = score @ a_norm.T  # (n_windows, N)
    return score + alpha * child_evidence


# ---------------------------------------------------------------------------
# Qualitative summary
# ---------------------------------------------------------------------------
def localization_report(score, true_mask, ks=(1, 3, 5)):
    """Light, qualitative hit@k summary over the truly anomalous windows.

    For every window that contains at least one anomalous variable, we check
    whether a *truly* anomalous variable appears among the top-k highest scoring
    variables. This is a sanity signal for the report, not a benchmark metric.

    Args:
        score: Per-variable attribution, shape (n_windows, N).
        true_mask: Per-window variable ground truth (bool), shape (n_windows, N).
        ks: Which top-k cutoffs to report.

    Returns:
        dict mapping ``"hit@k"`` -> fraction in [0, 1], plus ``"n_windows"``.
    """
    anomalous_windows = np.where(true_mask.any(axis=1))[0]
    out = {"n_windows": int(anomalous_windows.size)}
    if anomalous_windows.size == 0:
        for k in ks:
            out[f"hit@{k}"] = float("nan")
        return out

    ranked = np.argsort(-score[anomalous_windows], axis=1)  # high score first
    for k in ks:
        topk = ranked[:, :k]
        hits = [true_mask[w, topk[i]].any() for i, w in enumerate(anomalous_windows)]
        out[f"hit@{k}"] = float(np.mean(hits))
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_error_heatmap(score, true_mask, out_path, title="Per-variable anomaly score",
                       max_windows=400):
    """Heatmap of the per-variable attribution next to the ground truth.

    Top panel: predicted score (variables x windows). Bottom panel: the true
    per-window variable mask. Reading them together shows whether the bright
    spots in the prediction land on the truly anomalous variables.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting")

    if score.shape[0] > max_windows:
        score = score[:max_windows]
        true_mask = true_mask[:max_windows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    im = axes[0].imshow(score.T, aspect="auto", cmap="magma", interpolation="nearest")
    axes[0].set_title(title)
    axes[0].set_ylabel("variable")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.01, label="score")

    axes[1].imshow(true_mask.T, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[1].set_title("Ground-truth anomalous variables")
    axes[1].set_ylabel("variable")
    axes[1].set_xlabel("test window")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_window_case(score, true_mask, window_idx, out_path, top_k=10):
    """Bar chart of the per-variable score for a single window (a case study).

    Bars for truly anomalous variables are highlighted, so it is immediately
    visible whether the attribution ranks the right variable(s) at the top.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting")

    s = score[window_idx]
    truth = true_mask[window_idx]
    order = np.argsort(-s)[:top_k]
    colors = ["#d62728" if truth[v] else "#1f77b4" for v in order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(order)), s[order], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(v) for v in order])
    ax.set_xlabel("variable index (ranked by score)")
    ax.set_ylabel("anomaly score")
    ax.set_title(f"Window {window_idx}: red = truly anomalous variable")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def pick_example_window(true_mask):
    """Return the index of a representative anomalous window (or 0 if none)."""
    anomalous = np.where(true_mask.any(axis=1))[0]
    if anomalous.size == 0:
        return 0
    # Prefer a window with a single anomalous variable -> cleanest case study.
    counts = true_mask[anomalous].sum(axis=1)
    return int(anomalous[np.argmin(counts)])


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------
def run(result_dir, data_dir, anomaly_type, factor, win_size, input_step,
        out_dir, alpha=0.5, step=1):
    """Load artifacts, compute both attributions, write figures and a summary.

    Returns a dict with the two report summaries so callers (e.g. a harness)
    can aggregate them.
    """
    art = load_localization_artifacts(result_dir, data_dir, anomaly_type, factor)
    error, graph = art["error"], art["graph"]
    true_mask = window_var_labels(
        art["var_labels"], win_size, input_step, step=step, n_windows=error.shape[0]
    )

    direct = localize_direct(error, normalize=True)
    causal = localize_causal(error, graph, alpha=alpha, normalize=True)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{anomaly_type}_factor{factor}"
    summaries = {}
    if _HAS_MPL:
        plot_error_heatmap(direct, true_mask,
                           os.path.join(out_dir, f"{tag}_direct_heatmap.png"),
                           title=f"Direct attribution - {tag}")
        plot_error_heatmap(causal, true_mask,
                           os.path.join(out_dir, f"{tag}_causal_heatmap.png"),
                           title=f"Causal-propagated attribution - {tag}")
        w = pick_example_window(true_mask)
        plot_window_case(direct, true_mask, w,
                         os.path.join(out_dir, f"{tag}_direct_window{w}.png"))
        plot_window_case(causal, true_mask, w,
                         os.path.join(out_dir, f"{tag}_causal_window{w}.png"))

    summaries["direct"] = localization_report(direct, true_mask)
    summaries["causal"] = localization_report(causal, true_mask)

    print(f"[{tag}] direct  : {summaries['direct']}")
    print(f"[{tag}] causal  : {summaries['causal']}")
    return summaries


def _parse_args():
    p = argparse.ArgumentParser(description="CAROTS variable-level localization (Step 2).")
    p.add_argument("--result-dir", required=True,
                   help="Folder with per_variable_cd_error.npy and causality_matrix.npy.")
    p.add_argument("--data-dir", required=True,
                   help="Dataset folder with the *_varlabels.npy files.")
    p.add_argument("--anomaly-type", default="point_global",
                   help="Anomaly type, matching the test file name.")
    p.add_argument("--factor", default="2.0",
                   help="Factor in the test file name (use 'None' for collective_global).")
    p.add_argument("--win-size", type=int, default=10, help="Inference window size.")
    p.add_argument("--input-step", type=int, default=9,
                   help="CUTS_PLUS.INPUT_STEP used at inference.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Causal-propagation weight for the 'causal' attribution.")
    p.add_argument("--out-dir", default="localization_figures",
                   help="Where to write the figures.")
    return p.parse_args()


def main():
    args = _parse_args()
    run(
        result_dir=args.result_dir,
        data_dir=args.data_dir,
        anomaly_type=args.anomaly_type,
        factor=args.factor,
        win_size=args.win_size,
        input_step=args.input_step,
        out_dir=args.out_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
