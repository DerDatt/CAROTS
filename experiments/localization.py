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

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - metrics degrade gracefully
    _HAS_SKLEARN = False


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


def localize_direct(error, normalize=True, two_sided=False):
    """``direct`` attribution: the per-variable error itself.

    Args:
        error: Per-variable error matrix, shape (n_windows, N).
        normalize: If True, robust-zscore each variable first (recommended).
        two_sided: If True, score the *magnitude* of the deviation from the
            variable's typical error instead of its signed value. This matters
            for anomalies that make a variable **easier** to forecast: a
            collective-global level shift replaces a noisy segment with a
            smooth one, so the residual drops far below the variable's median
            and a signed score ranks the true culprit as maximally *normal*.
            Requires ``normalize=True`` to be meaningful.

    Returns:
        Score matrix (n_windows, N); higher means "more likely the culprit".
    """
    if not normalize:
        return error
    score = robust_zscore(error)
    return np.abs(score) if two_sided else score


def _row_normalize(graph):
    """Row-normalize the adjacency so each row averages over a node's children."""
    out_degree = graph.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1.0  # avoid div-by-zero for childless nodes
    return graph / out_degree


def localize_causal(error, graph, alpha=0.5, normalize=True, two_sided=False):
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
        two_sided: Score deviation magnitude instead of signed deviation; see
            :func:`localize_direct`.

    Returns:
        Score matrix (n_windows, N).
    """
    score = localize_direct(error, normalize=normalize, two_sided=two_sided)
    # Drop self-loops on the diagonal so a variable does not "propagate to
    # itself"; the diagonal of CAROTS' causality matrix is typically 1.
    children = graph.copy()
    np.fill_diagonal(children, 0.0)
    a_norm = _row_normalize(children)
    child_evidence = score @ a_norm.T  # (n_windows, N)
    return score + alpha * child_evidence


# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------
def _random_hit_at_k(n_vars, n_true, k):
    """Expected hit@k of a random ranking: 1 - P(no true variable in top-k).

    Drawing k of ``n_vars`` variables without replacement, the probability that
    none of the ``n_true`` anomalous ones is among them is the product of
    (n_vars - n_true - i) / (n_vars - i) for i < k.
    """
    if n_true <= 0 or k <= 0:
        return 0.0
    k = min(k, n_vars)
    p_miss = 1.0
    for i in range(k):
        num = n_vars - n_true - i
        if num <= 0:
            return 1.0
        p_miss *= num / (n_vars - i)
    return 1.0 - p_miss


def _mean_within_window_auroc(score, true_mask):
    """Mean of the per-window AUROC, i.e. ranking variables *inside* a window.

    This is the metric localization actually needs, and it can differ sharply
    from the pooled ``var_auroc``. Pooling all (window, variable) pairs lets a
    "loud" window - one where every variable happens to have a high score -
    outrank a quiet one, which inflates the number even if the culprits are not
    separated from their own window's normal variables.

    Computed from ranks (Mann-Whitney form) so no per-window sklearn call is
    needed: with ``m`` positives and ``n`` negatives,
    ``AUROC = (sum of positive ranks - m(m+1)/2) / (m*n)``.
    """
    n_pos = true_mask.sum(axis=1)
    n_neg = true_mask.shape[1] - n_pos
    usable = (n_pos > 0) & (n_neg > 0)
    if not usable.any():
        return float("nan")

    s, t, n_pos, n_neg = score[usable], true_mask[usable], n_pos[usable], n_neg[usable]
    order = np.argsort(s, axis=1)                      # ascending
    ranks = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, s.shape[1] + 1)  # 1-based ranks
    pos_rank_sum = (ranks * t).sum(axis=1)
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(np.mean(auroc))


def localization_report(score, true_mask, ks=(1, 3, 5)):
    """Quantitative variable-level localization metrics.

    Everything is computed **on the truly anomalous windows only**: the question
    Step 2 answers is "given that this window is anomalous, which variable is to
    blame?", so normal windows would only dilute the numbers.

    Metrics:
        ``auroc_within`` - mean per-window AUROC: can the culprits be separated
            from the *other variables of the same window*? This is the metric
            localization actually needs; read it against 0.5.
        ``var_auroc`` / ``var_ap``  - pool all (window, variable) pairs of the
            anomalous windows and score the attribution as a binary ranking
            problem against the per-variable ground truth. ``var_ap`` should be
            read against ``random_ap`` (the anomalous-variable base rate).
            Beware: pooling mixes in *between*-window variation, so these can
            look strong while ``auroc_within`` and ``hit@k`` sit at chance.
        ``mrr``  - mean reciprocal rank of the *first* truly anomalous variable.
            1.0 means the culprit is always ranked first.
        ``hit@k`` - fraction of anomalous windows whose top-k contains a truly
            anomalous variable, with ``random_hit@k`` as the chance level.

    Args:
        score: Per-variable attribution, shape (n_windows, N).
        true_mask: Per-window variable ground truth (bool), shape (n_windows, N).
        ks: Which top-k cutoffs to report.

    Returns:
        dict of metric name -> value, plus ``n_windows`` and ``n_vars``.
    """
    n_vars = int(true_mask.shape[1])
    anomalous_windows = np.where(true_mask.any(axis=1))[0]
    out = {"n_windows": int(anomalous_windows.size), "n_vars": n_vars}
    if anomalous_windows.size == 0:
        for key in ("auroc_within", "var_auroc", "var_ap", "random_ap", "mrr",
                    "random_mrr"):
            out[key] = float("nan")
        for k in ks:
            out[f"hit@{k}"] = float("nan")
            out[f"random_hit@{k}"] = float("nan")
        return out

    s = score[anomalous_windows]
    t = true_mask[anomalous_windows]

    out["auroc_within"] = _mean_within_window_auroc(s, t)

    flat_score, flat_true = s.ravel(), t.ravel().astype(int)
    base_rate = float(flat_true.mean())
    if _HAS_SKLEARN and 0 < flat_true.sum() < flat_true.size:
        out["var_auroc"] = float(roc_auc_score(flat_true, flat_score))
        out["var_ap"] = float(average_precision_score(flat_true, flat_score))
    else:
        out["var_auroc"] = float("nan")
        out["var_ap"] = float("nan")
    out["random_ap"] = base_rate

    ranked = np.argsort(-s, axis=1)  # high score first
    ranked_truth = np.take_along_axis(t, ranked, axis=1)
    # Rank (1-based) of the first truly anomalous variable in each window.
    first_hit = np.argmax(ranked_truth, axis=1) + 1
    out["mrr"] = float(np.mean(1.0 / first_hit))
    n_true_per_window = t.sum(axis=1)
    # Chance level: with m of N variables anomalous, the expected reciprocal rank
    # of a random permutation is well approximated by averaging 1/rank over the
    # (N+1)/(m+1) expected first-hit position.
    out["random_mrr"] = float(np.mean((n_true_per_window + 1) / (n_vars + 1)))

    for k in ks:
        out[f"hit@{k}"] = float(ranked_truth[:, :k].any(axis=1).mean())
        out[f"random_hit@{k}"] = float(np.mean(
            [_random_hit_at_k(n_vars, int(m), k) for m in n_true_per_window]
        ))
    return out


def alpha_sweep(error, graph, true_mask, alphas=(0.0, 0.25, 0.5, 1.0, 2.0),
                normalize=True, ks=(1, 3), two_sided_modes=(False, True)):
    """Does causal propagation actually help? Sweep the propagation weight.

    ``alpha=0`` is exactly the ``direct`` attribution, so this isolates the
    contribution of the propagation term. If the metrics are flat across alpha,
    the propagation adds nothing on this dataset - which is a result worth
    reporting rather than hiding.

    The sweep is repeated for the signed and the two-sided score (see
    :func:`localize_direct`), because which one wins depends on the anomaly
    type.

    Returns:
        list of dicts, one per (score mode, alpha), each containing
        ``score_mode``, ``alpha`` and the metrics from
        :func:`localization_report`.
    """
    rows = []
    for two_sided in two_sided_modes:
        for alpha in alphas:
            score = localize_causal(error, graph, alpha=alpha,
                                    normalize=normalize, two_sided=two_sided)
            rows.append({"score_mode": "two_sided" if two_sided else "signed",
                         "alpha": float(alpha),
                         **localization_report(score, true_mask, ks=ks)})
    return rows


def leakage_report(score, true_mask, graph):
    """How far does the forecasting error spread from the true culprit?

    Splits the variables of every anomalous window into three groups and returns
    their mean attribution score:

        ``culprit``  - the truly anomalous variables,
        ``children`` - their causal children (which inherit a corrupted input),
        ``other``    - everything else.

    This is the diagnostic behind the causal propagation: it only has something
    to work with if ``children`` sits clearly above ``other``. If children look
    like unrelated variables, there is no leakage to undo.
    """
    anomalous_windows = np.where(true_mask.any(axis=1))[0]
    out = {"culprit": float("nan"), "children": float("nan"), "other": float("nan")}
    if anomalous_windows.size == 0:
        return out

    adjacency = graph.copy().astype(float)
    np.fill_diagonal(adjacency, 0.0)
    means = {"culprit": [], "children": [], "other": []}
    for w in anomalous_windows:
        culprit = true_mask[w]
        # A variable is a "child" if any culprit points at it (and is not one).
        child = (adjacency[culprit].sum(axis=0) > 0) & ~culprit
        other = ~culprit & ~child
        for key, mask in (("culprit", culprit), ("children", child), ("other", other)):
            if mask.any():
                means[key].append(float(score[w, mask].mean()))
    for key, values in means.items():
        if values:
            out[key] = float(np.mean(values))
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_error_heatmap(score, true_mask, out_path, title="Per-variable anomaly score",
                       max_windows=400, clip_percentile=99.0):
    """Heatmap of the per-variable attribution next to the ground truth.

    Top panel: predicted score (variables x windows). Bottom panel: the true
    per-window variable mask. Reading them together shows whether the bright
    spots in the prediction land on the truly anomalous variables.

    The colour range is clipped to ``clip_percentile`` of the scores. Without
    this, a single extreme window (forecasting errors span several orders of
    magnitude) saturates the scale and renders everything else black.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting")

    if score.shape[0] > max_windows:
        score = score[:max_windows]
        true_mask = true_mask[:max_windows]

    finite = score[np.isfinite(score)]
    vmin, vmax = None, None
    if finite.size:
        vmin = float(np.percentile(finite, 100 - clip_percentile))
        vmax = float(np.percentile(finite, clip_percentile))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max()) or None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    im = axes[0].imshow(score.T, aspect="auto", cmap="magma", interpolation="nearest",
                        vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title}\n(colour clipped at the {clip_percentile:g}th percentile)")
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


def plot_alpha_sweep(rows, out_path, title="Effect of causal propagation",
                     metrics=("auroc_within", "mrr", "hit@1")):
    """Localization quality against the propagation weight alpha, per score mode.

    One panel per score mode (signed / two-sided). A flat line means the causal
    propagation neither helps nor hurts; the dotted line is the chance level of
    the corresponding metric.
    """
    if not _HAS_MPL or not rows:
        return None

    modes = sorted({r.get("score_mode", "signed") for r in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        subset = [r for r in rows if r.get("score_mode", "signed") == mode]
        alphas = [r["alpha"] for r in subset]
        for metric in metrics:
            if metric not in subset[0]:
                continue
            line, = ax.plot(alphas, [r[metric] for r in subset],
                            marker="o", label=metric)
            chance = "random_ap" if metric == "var_ap" else f"random_{metric}"
            if metric in ("var_auroc", "auroc_within"):
                ax.axhline(0.5, ls=":", lw=1, color=line.get_color())
            elif chance in subset[0] and np.isfinite(subset[0][chance]):
                ax.axhline(subset[0][chance], ls=":", lw=1, color=line.get_color())
        ax.set_xlabel(r"propagation weight $\alpha$   ($\alpha=0$: direct)")
        ax.set_title(f"{mode} score")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("score  (dotted = chance level)")
    axes[0][-1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def pick_example_window(true_mask, score=None):
    """Return the index of a *representative* anomalous window (or 0 if none).

    Picking by ground truth alone is not enough: when every anomalous window has
    exactly one culprit, ``argmin`` over the culprit count silently returns the
    first anomalous window, which is often the least typical one (the start of an
    injected segment, before the disturbance has taken effect). A case-study
    figure built from it then misrepresents the run in both directions.

    When ``score`` is given we instead rank the culprit within each anomalous
    window and return the window whose culprit rank is the **median** - the
    honest middle of the distribution rather than a lucky or unlucky extreme.
    """
    anomalous = np.where(true_mask.any(axis=1))[0]
    if anomalous.size == 0:
        return 0

    counts = true_mask[anomalous].sum(axis=1)
    candidates = anomalous[counts == counts.min()]
    if score is None:
        return int(candidates[0])

    s = score[candidates]
    t = true_mask[candidates]
    ranked_truth = np.take_along_axis(t, np.argsort(-s, axis=1), axis=1)
    best_rank = np.argmax(ranked_truth, axis=1)
    # argsort is stable, so ties resolve to the earliest window at that rank.
    median_position = np.argsort(best_rank, kind="stable")[len(best_rank) // 2]
    return int(candidates[median_position])


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def _format_metrics(d):
    """One-line rendering of a metrics dict for console output."""
    keys = ("auroc_within", "var_auroc", "var_ap", "random_ap", "mrr", "hit@1",
            "random_hit@1", "n_windows")
    parts = []
    for k in keys:
        v = d.get(k)
        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)


def run(result_dir, data_dir, anomaly_type, factor, win_size, input_step,
        out_dir, alpha=0.5, step=1, alphas=DEFAULT_ALPHAS,
        two_sided_modes=(False, True)):
    """Load artifacts, evaluate every attribution, write figures and metrics.

    Evaluates the cross product of {direct, causal} x {signed, two-sided} so the
    two independent design choices - propagating along the graph, and scoring
    the magnitude instead of the sign of the deviation - can be judged
    separately.

    Returns:
        dict with one metrics entry per ``"<attribution>_<mode>"`` (e.g.
        ``"direct_signed"``), plus ``alpha_sweep`` and the ``leakage``
        diagnostic, so callers (e.g. the aggregator) can build a table.
    """
    art = load_localization_artifacts(result_dir, data_dir, anomaly_type, factor)
    error, graph = art["error"], art["graph"]
    true_mask = window_var_labels(
        art["var_labels"], win_size, input_step, step=step, n_windows=error.shape[0]
    )

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{anomaly_type}_factor{factor}"
    sweep = alpha_sweep(error, graph, true_mask, alphas=alphas,
                        two_sided_modes=two_sided_modes)
    summaries = {"alpha_sweep": sweep}

    # One window for all four figures so they stay comparable; chosen as the
    # median case of the plain signed attribution.
    example_window = pick_example_window(
        true_mask, localize_direct(error, normalize=True))
    for two_sided in two_sided_modes:
        mode = "two_sided" if two_sided else "signed"
        scores = {
            "direct": localize_direct(error, normalize=True, two_sided=two_sided),
            "causal": localize_causal(error, graph, alpha=alpha, normalize=True,
                                      two_sided=two_sided),
        }
        for attribution, score in scores.items():
            summaries[f"{attribution}_{mode}"] = localization_report(score, true_mask)
            if not _HAS_MPL:
                continue
            stem = f"{tag}_{attribution}_{mode}"
            plot_error_heatmap(
                score, true_mask, os.path.join(out_dir, f"{stem}_heatmap.png"),
                title=f"{attribution} attribution ({mode}) - {tag}")
            plot_window_case(
                score, true_mask, example_window,
                os.path.join(out_dir, f"{stem}_window{example_window}.png"))

    # The leakage diagnostic characterizes the data, not a scoring choice, so it
    # is computed once on the plain signed attribution.
    summaries["leakage"] = leakage_report(
        localize_direct(error, normalize=True), true_mask, graph)

    if _HAS_MPL:
        plot_alpha_sweep(sweep, os.path.join(out_dir, f"{tag}_alpha_sweep.png"),
                         title=f"Causal propagation - {tag}")

    for key in sorted(k for k in summaries if k not in ("alpha_sweep", "leakage")):
        print(f"[{tag}] {key:<18}: {_format_metrics(summaries[key])}")
    leak = summaries["leakage"]
    print(f"[{tag}] leakage (signed)  : culprit={leak['culprit']:.3f}  "
          f"children={leak['children']:.3f}  other={leak['other']:.3f}")
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
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                   help="Propagation weights to sweep (alpha=0 equals 'direct').")
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
        alphas=tuple(args.alphas),
    )


if __name__ == "__main__":
    main()
