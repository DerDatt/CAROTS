"""Build the small set of figures that carry the talk.

The per-run figures written by ``localization.py`` are diagnostic: one file per
(scenario, seed, anomaly type), which is the right granularity for checking a
run but the wrong one for a 15-minute presentation. This script aggregates the
CSVs that ``aggregate.py`` produces into a handful of slide-ready figures, each
making exactly one point.

Headline figures, driven purely by the aggregated CSVs:

    fig1_detection_robustness.png  Step 1: which broken assumption hurts?
    fig2_localization_position.png Step 2: is it really finding the culprit, or
                                   just always answering variable 0?
    fig2a_simple_example.png       one window, anomaly on var 0: culprit first
    fig2b_simple_hit.png           hit@1 only for toy_chain (anomaly on var 0)
    fig2c_simple_examples_grid.png four anomaly types, each one example window
    fig3_propagation_alpha.png     Step 2: when does the causal term help - and
                                   when does it actively destroy the answer?

Explanatory figures, which additionally need the raw ``.npy`` artifacts and the
datasets (pass ``--toy-root`` / ``--step1-root`` / ``--base-dir``):

    fig4_toy_system.png            what the toy causal chain looks like and where
                                   each control injects its anomaly
    fig5_propagation_mechanism.png one window, direct vs propagated: *why* the
                                   causal term points at the parent
    fig6_propagation_both.png      the alpha sweep for both point anomaly types,
                                   so fig3 is not a single-anomaly artifact
    fig7_highdim_paradox.png       how AUROC 0.95 and hit@1 = 0.00 coexist at
                                   p=128: the maximum of 118 noise variables
    fig8_detection_vs_localization.png  flawed training data hurts detection far
                                   more than it hurts attribution
    fig9_leakage_by_anomaly.png    the precondition for propagation: does the
                                   disturbance reach the culprit's children?

Usage:
    python -m experiments.plot_presentation --results-root results
    # also build the explanatory figures:
    python -m experiments.plot_presentation \\
        --results-root results --toy-root results \\
        --step1-root results --base-dir data

Figures always land in ``results/slides/`` unless ``--out-dir`` overrides that.
If results were copied into a nested export folder, point ``--results-root``
(and friends) at that folder instead of ``results/``.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import localization as L  # noqa: E402
from experiments.aggregate import VAR_INPUT_STEP, VAR_WIN_SIZE  # noqa: E402
from experiments.scenarios import SCENARIOS, TOY_SCENARIOS  # noqa: E402


def _true_mask(art):
    """Per-window variable ground truth, aligned exactly as aggregate.py does."""
    return L.window_var_labels(art["var_labels"], win_size=VAR_WIN_SIZE,
                               input_step=VAR_INPUT_STEP,
                               n_windows=art["error"].shape[0])

# Anomaly types in a fixed order, with the short labels used on the slides.
ANOMALY_LABELS = {
    "point_global": "point\nglobal",
    "point_contextual": "point\ncontextual",
    "collective_trend": "collective\ntrend",
    "collective_global": "collective\nglobal",
}
SCENARIO_LABELS = {
    "baseline": "baseline (clean)",
    "nocausal": "no causal structure",
    "nonstationary": "non-stationary",
    "contaminated": "contaminated training",
}
# The toys differ only in where the anomaly sits on the chain 0 -> 1 -> 2.
POSITION_LABELS = {
    "toy_chain": "root (var 0)",
    "toy_chain_mid": "middle (var 1)",
    "toy_chain_leaf": "leaf (var 2)",
}
POSITION_COLORS = {
    "toy_chain": "#1b7837",
    "toy_chain_mid": "#762a83",
    "toy_chain_leaf": "#d95f02",
}


def _bar_handles(ax):
    """The bar containers only.

    ``ax.containers`` also holds the ErrorbarContainers created by ``yerr=``, and
    an ``axhline`` adds a further line handle. Letting matplotlib pick handles
    automatically therefore pairs the labels with the wrong colours.
    """
    return [c for c in ax.containers if isinstance(c, matplotlib.container.BarContainer)]


def _style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def fig_detection_robustness(summary_by_seed, out_path):
    """Step 1: AUROC per anomaly type, one bar group per training-data flaw."""
    df = summary_by_seed[summary_by_seed["scenario"].isin(SCENARIO_LABELS)]
    order = [a for a in ANOMALY_LABELS if a in set(df["anomaly_type"])]
    scenarios = [s for s in SCENARIO_LABELS if s in set(df["scenario"])]

    mean = df.pivot_table(index="anomaly_type", columns="scenario",
                          values="AUROC_mean").reindex(order)[scenarios]
    std = df.pivot_table(index="anomaly_type", columns="scenario",
                         values="AUROC_std").reindex(order)[scenarios]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    mean.plot(kind="bar", yerr=std, capsize=4, ax=ax, width=0.78,
              color=["#4575b4", "#fdae61", "#d73027", "#7570b3"])
    ax.set_xticklabels([ANOMALY_LABELS[a] for a in order], rotation=0)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("")
    ax.set_ylim(0.5, 1.06)
    ax.set_title("Detection: which violated assumption actually hurts?")
    ax.legend(_bar_handles(ax), [SCENARIO_LABELS[s] for s in scenarios],
              loc="upper left", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_localization_position(metrics, out_path):
    """Step 2 control: does the attribution track the culprit's position?"""
    df = metrics[(metrics["attribution"] == "direct")
                 & (metrics["score_mode"] == "signed")
                 & (metrics["scenario"].isin(POSITION_LABELS))]
    order = [a for a in ANOMALY_LABELS if a in set(df["anomaly_type"])]
    positions = [p for p in POSITION_LABELS if p in set(df["scenario"])]

    grouped = df.groupby(["anomaly_type", "scenario"])["hit@1"]
    mean = grouped.mean().unstack("scenario").reindex(order)[positions]
    std = grouped.std().unstack("scenario").reindex(order)[positions]
    chance = df["random_hit@1"].mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    mean.plot(kind="bar", yerr=std, capsize=4, ax=ax, width=0.72,
              color=[POSITION_COLORS[p] for p in positions])
    ax.set_xticklabels([ANOMALY_LABELS[a] for a in order], rotation=0)
    ax.set_ylabel("hit@1  (culprit ranked first)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    handles = _bar_handles(ax)
    ax.axhline(chance, color="black", lw=1.4, ls="--")
    ax.text(0.99, chance + 0.02, f"chance = {chance:.2f}", fontsize=11,
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_title("Localization does not just always answer variable 0")
    ax.legend(handles, [POSITION_LABELS[p] for p in positions],
              title="anomaly sits on", loc="center right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_propagation_alpha(sweep, out_path, anomaly_type="point_global"):
    """Step 2: the causal term helps upstream and destroys the answer downstream."""
    df = sweep[(sweep["score_mode"] == "signed")
               & (sweep["anomaly_type"] == anomaly_type)
               & (sweep["scenario"].isin(POSITION_LABELS))]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for scenario in [p for p in POSITION_LABELS if p in set(df["scenario"])]:
        sub = df[df["scenario"] == scenario].groupby("alpha")["hit@1"]
        mean, std = sub.mean(), sub.std()
        ax.plot(mean.index, mean.values, marker="o", lw=2.2,
                color=POSITION_COLORS[scenario], label=POSITION_LABELS[scenario])
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.18,
                        color=POSITION_COLORS[scenario])

    chance = df["random_hit@1"].mean()
    ax.axhline(chance, color="black", lw=1.4, ls="--")
    ax.text(0.99, chance + 0.02, f"chance = {chance:.2f}", fontsize=11,
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"causal propagation weight  $\alpha$      ($\alpha=0$: no propagation)")
    ax.set_ylabel("hit@1  (culprit ranked first)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Causal propagation helps only if the culprit is upstream")
    ax.legend(title="anomaly sits on", loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_toy_system(out_path, base_dir="data"):
    """Draw the toy causal chain and where each control injects its anomaly."""
    meta_path = os.path.join(base_dir, "VAR_toy_chain", "toy_meta.npz")
    if not os.path.exists(meta_path):
        print(f"  missing {meta_path} - skipping fig4")
        return None
    meta = np.load(meta_path, allow_pickle=True)
    chain_len = int(meta["chain_len"])
    n_distractors = int(meta["n_distractors"])
    couple = float(meta["couple"])

    fig, ax = plt.subplots(figsize=(10, 4.4))
    chain_y, distractor_y = 0.62, 0.20
    xs = np.linspace(0.12, 0.62, chain_len)

    # Which control targets which chain position.
    target_of = {v: name for name, v in TOY_SCENARIOS.items()}
    label_of = {"toy_chain": "root", "toy_chain_mid": "middle",
                "toy_chain_leaf": "leaf"}

    for i, x in enumerate(xs):
        colour = POSITION_COLORS.get(target_of.get(i), "#999999")
        ax.add_patch(plt.Circle((x, chain_y), 0.045, color=colour, zorder=3))
        ax.text(x, chain_y, str(i), ha="center", va="center", color="white",
                fontsize=14, fontweight="bold", zorder=4)
        if i in target_of:
            ax.annotate(label_of[target_of[i]], xy=(x, chain_y + 0.055),
                        xytext=(x, chain_y + 0.20), ha="center", fontsize=12,
                        color=colour, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=2))
        if i + 1 < chain_len:
            ax.annotate("", xy=(xs[i + 1] - 0.05, chain_y),
                        xytext=(x + 0.05, chain_y),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333"))
            ax.text((x + xs[i + 1]) / 2, chain_y - 0.075, f"{couple:g}",
                    ha="center", fontsize=11, color="#333333")

    for j in range(n_distractors):
        x = xs[j] if n_distractors <= chain_len else 0.12 + 0.25 * j
        ax.add_patch(plt.Circle((x, distractor_y), 0.045, color="#bbbbbb", zorder=3))
        ax.text(x, distractor_y, str(chain_len + j), ha="center", va="center",
                color="white", fontsize=14, fontweight="bold", zorder=4)
    ax.text(xs[min(n_distractors, chain_len - 1)] + 0.09, distractor_y,
            "distractors: no causal links",
            va="center", fontsize=12, color="#555555")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.02, 0.95)
    ax.axis("off")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _load_toy(toy_root, scenario, seed, anomaly, factor, base_dir):
    result_dir = os.path.join(toy_root, scenario, f"seed{seed}",
                              f"VAR_{anomaly}_factor{factor}")
    data_dir = os.path.join(base_dir, SCENARIOS[scenario])
    if not os.path.exists(os.path.join(result_dir, "per_variable_cd_error.npy")):
        return None
    art = L.load_localization_artifacts(result_dir, data_dir, anomaly, factor)
    art["true_mask"] = _true_mask(art)
    return art


def fig_simple_localization_hit(metrics, out_path, scenario="toy_chain"):
    """hit@1 per anomaly type for one toy control (default: anomaly on var 0)."""
    df = metrics[(metrics["scenario"] == scenario)
                 & (metrics["attribution"] == "direct")
                 & (metrics["score_mode"] == "signed")]
    if df.empty:
        return None
    order = [a for a in ANOMALY_LABELS if a in set(df["anomaly_type"])]
    grouped = df.groupby("anomaly_type")["hit@1"]
    mean, std = grouped.mean().reindex(order), grouped.std().reindex(order)
    chance = float(df["random_hit@1"].iloc[0])
    culprit = TOY_SCENARIOS.get(scenario, 0)
    colour = POSITION_COLORS.get(scenario, "#1b7837")
    where = POSITION_LABELS.get(scenario, f"var {culprit}")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(order))
    ax.bar(x, mean.values, yerr=std.values, capsize=5, width=0.65,
           color=colour, ecolor="black")
    ax.axhline(chance, color="black", lw=1.5, ls="--")
    ax.text(len(order) - 0.55, chance + 0.03, f"chance = {chance:.2f}",
            fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ANOMALY_LABELS[a] for a in order])
    ax.set_ylabel("hit@1  (true culprit ranked first)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Localization works: anomaly on {where}")
    for i, (m, s) in enumerate(zip(mean.values, std.values)):
        ax.text(i, m + (0 if np.isnan(s) else s) + 0.02, f"{m:.2f}",
                ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_simple_example_window(out_path, toy_root, base_dir, seed=2,
                              scenario="toy_chain",
                              anomaly="collective_trend", factor="3.0"):
    """One representative window: the true culprit is ranked first."""
    art = _load_toy(toy_root, scenario, seed, anomaly, factor, base_dir)
    if art is None:
        print(f"  missing {scenario} artifacts - skipping simple example window")
        return None
    score = L.localize_direct(art["error"], normalize=True)
    true_mask = art["true_mask"]
    window = L.pick_example_window(true_mask, score)
    s = score[window]
    order = np.argsort(-s)
    culprit = int(np.argmax(true_mask[window]))
    rep = L.localization_report(score, true_mask)
    colour = POSITION_COLORS.get(scenario, "#d73027")

    colours = [colour if v == culprit else "#bbbbbb" for v in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(range(len(order)), s[order], color=colours)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"var {v}" for v in order])
    ax.set_ylabel("anomaly score")
    ax.set_xlabel("variables, ranked by score")
    ax.set_title(
        f"Example window ({anomaly.replace('_', ' ')}): "
        f"var {culprit} ranked first\n"
        f"over all anomalous windows: hit@1 = {rep['hit@1']:.2f}  "
        f"(chance {rep['random_hit@1']:.2f})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (colour, "#bbbbbb")]
    ax.legend(handles, [f"true culprit (var {culprit})", "other variables"],
              loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_simple_examples_grid(out_path, toy_root, base_dir, seed=2,
                             scenario="toy_chain"):
    """Four example windows (one per anomaly type) for one injection target."""
    cases = [("point_global", "3.0"), ("point_contextual", "3.0"),
             ("collective_trend", "3.0"), ("collective_global", "None")]
    culprit_expected = TOY_SCENARIOS.get(scenario, 0)
    colour = POSITION_COLORS.get(scenario, "#d73027")

    # Wide layout: 1x4 keeps the figure short enough for a slide.
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False)
    for ax, (anomaly, factor) in zip(axes, cases):
        art = _load_toy(toy_root, scenario, seed, anomaly, factor, base_dir)
        if art is None:
            ax.set_visible(False)
            continue
        score = L.localize_direct(art["error"], normalize=True)
        true_mask = art["true_mask"]
        window = L.pick_example_window(true_mask, score)
        s = score[window]
        order = np.argsort(-s)
        culprit = int(np.argmax(true_mask[window]))
        rep = L.localization_report(score, true_mask)
        colours = [colour if v == culprit else "#bbbbbb" for v in order]
        ax.bar(range(len(order)), s[order], color=colours)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(v) for v in order])
        ax.set_title(f"{anomaly.replace('_', ' ')}\n"
                     f"hit@1 = {rep['hit@1']:.2f}  (chance {rep['random_hit@1']:.2f})",
                     fontsize=11)
        ax.set_xlabel("variable rank")
    axes[0].set_ylabel("anomaly score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (colour, "#bbbbbb")]
    fig.legend(handles, [f"true culprit (var {culprit_expected})",
                         "other variables"],
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_propagation_mechanism(out_path, toy_root, base_dir, alpha=1.0,
                              scenario="toy_chain_leaf",
                              anomaly="point_global", factor="3.0", seed=2):
    """One window, direct vs propagated, to show *why* propagation misfires.

    The window is chosen as the median case of the direct attribution, not
    hand-picked, and both panels report the aggregate hit@1 so the single example
    cannot be mistaken for the whole story.
    """
    art = _load_toy(toy_root, scenario, seed, anomaly, factor, base_dir)
    if art is None:
        print("  missing toy artifacts - skipping fig5")
        return None
    error, graph, true_mask = art["error"], art["graph"], art["true_mask"]

    direct = L.localize_direct(error, normalize=True)
    causal = L.localize_causal(error, graph, alpha=alpha, normalize=True)
    window = L.pick_example_window(true_mask, direct)
    culprit = int(np.argmax(true_mask[window]))
    parents = np.where(graph[:, culprit] > 0)[0]

    hit_direct = L.localization_report(direct, true_mask)["hit@1"]
    hit_causal = L.localization_report(causal, true_mask)["hit@1"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, score, name, hit in ((axes[0], direct, "direct", hit_direct),
                                 (axes[1], causal,
                                  rf"with propagation ($\alpha={alpha:g}$)",
                                  hit_causal)):
        s = score[window]
        order = np.argsort(-s)
        colours = ["#d73027" if true_mask[window, v] else
                   ("#4575b4" if v in parents else "#bbbbbb") for v in order]
        ax.bar(range(len(order)), s[order], color=colours)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(v) for v in order])
        ax.set_xlabel("variable, ranked by score")
        ax.set_title(f"{name}\nhit@1 over all windows = {hit:.3f}")
    axes[0].set_ylabel("anomaly score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ("#d73027", "#4575b4", "#bbbbbb")]
    axes[1].legend(handles, [f"true culprit (var {culprit})",
                             "its causal parent(s)", "other variables"],
                   loc="upper right", framealpha=0.95)
    fig.suptitle("Propagation credits the parent instead of the leaf culprit",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_propagation_both(sweep, out_path):
    """The alpha sweep for both point anomaly types, side by side."""
    types = [a for a in ("point_global", "point_contextual")
             if a in set(sweep["anomaly_type"])]
    if not types:
        return None
    fig, axes = plt.subplots(1, len(types), figsize=(6.2 * len(types), 5.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, anomaly in zip(axes, types):
        df = sweep[(sweep["score_mode"] == "signed")
                   & (sweep["anomaly_type"] == anomaly)
                   & (sweep["scenario"].isin(POSITION_LABELS))]
        for scenario in [p for p in POSITION_LABELS if p in set(df["scenario"])]:
            sub = df[df["scenario"] == scenario].groupby("alpha")["hit@1"]
            mean, std = sub.mean(), sub.std()
            ax.plot(mean.index, mean.values, marker="o", lw=2.2,
                    color=POSITION_COLORS[scenario],
                    label=POSITION_LABELS[scenario])
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.18,
                            color=POSITION_COLORS[scenario])
        ax.axhline(df["random_hit@1"].mean(), color="black", lw=1.4, ls="--")
        ax.set_title(anomaly.replace("_", " "))
        ax.set_xlabel(r"propagation weight $\alpha$")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("hit@1  (culprit ranked first)")
    axes[0].legend(title="anomaly sits on", loc="lower left", framealpha=0.95)
    fig.suptitle("The same collapse for both point anomaly types", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_highdim_paradox(out_path, step1_root, base_dir, scenario="baseline",
                        anomaly="point_global", factor="2.0", seed=2):
    """Why a within-window AUROC of 0.95 can coexist with hit@1 = 0.00.

    Left: the culprit scores sit clearly above *typical* normal variables, which
    is what the AUROC measures. Right: but they lose to the per-window *maximum*
    of the ~118 normal variables, which is what hit@1 measures.
    """
    result_dir = os.path.join(step1_root, scenario, f"seed{seed}",
                              f"VAR_{anomaly}_factor{factor}")
    data_dir = os.path.join(base_dir, SCENARIOS[scenario])
    if not os.path.exists(os.path.join(result_dir, "per_variable_cd_error.npy")):
        print(f"  missing {result_dir} - skipping fig7")
        return None
    art = L.load_localization_artifacts(result_dir, data_dir, anomaly, factor)
    true_mask = _true_mask(art)
    score = L.localize_direct(art["error"], normalize=True)

    rows = np.where(true_mask.any(axis=1))[0]
    s, t = score[rows], true_mask[rows]
    culprit_scores = s[t]
    normal_scores = s[~t]
    per_window_max_normal = np.where(t, -np.inf, s).max(axis=1)
    per_window_max_culprit = np.where(t, s, -np.inf).max(axis=1)

    rep = L.localization_report(score, true_mask)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    bins = np.linspace(0, np.percentile(normal_scores, 99.9), 60)
    axes[0].hist(normal_scores, bins=bins, color="#bbbbbb", label="normal variables")
    axes[0].hist(culprit_scores, bins=bins, color="#d73027", alpha=0.85,
                 label="true culprits")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("anomaly score")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title(f"Typical normal variable loses to the culprit\n"
                      f"within-window AUROC = {rep['auroc_within']:.3f}")
    axes[0].legend(framealpha=0.95)

    lo = min(per_window_max_normal.min(), per_window_max_culprit.min())
    hi = np.percentile(np.r_[per_window_max_normal, per_window_max_culprit], 99.5)
    bins2 = np.linspace(lo, hi, 50)
    axes[1].hist(per_window_max_normal, bins=bins2, color="#4575b4", alpha=0.8,
                 label="best of the normal variables")
    axes[1].hist(per_window_max_culprit, bins=bins2, color="#d73027", alpha=0.75,
                 label="best of the culprits")
    axes[1].set_xlabel("per-window maximum score")
    axes[1].set_ylabel("number of windows")
    axes[1].set_title(f"but the best of ~{int((~t[0]).sum())} normals wins\n"
                      f"hit@1 = {rep['hit@1']:.3f},  hit@5 = {rep['hit@5']:.3f} "
                      f"(chance {rep['random_hit@5']:.2f})")
    axes[1].legend(framealpha=0.95)

    fig.suptitle(f"At p={s.shape[1]} a good AUROC does not make top-1 usable",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_detection_vs_localization(out_path, summary_by_seed, step1_root, base_dir,
                                  seed=2, factors=None):
    """Detection degrades under flawed training data; attribution barely does.

    Detection has to judge a window's *total* score against what training led it
    to expect - exactly what a shifted or contaminated training set corrupts.
    Attribution only ranks variables against each other inside one window, so a
    miscalibrated overall error level largely cancels out.
    """
    factors = factors or {"point_global": "2.0", "point_contextual": "2.0",
                          "collective_trend": "2.0", "collective_global": "None"}
    scenarios = [s for s in SCENARIO_LABELS]
    rows = []
    for scenario in scenarios:
        for anomaly, factor in factors.items():
            result_dir = os.path.join(step1_root, scenario, f"seed{seed}",
                                      f"VAR_{anomaly}_factor{factor}")
            if not os.path.exists(os.path.join(result_dir,
                                               "per_variable_cd_error.npy")):
                continue
            art = L.load_localization_artifacts(
                result_dir, os.path.join(base_dir, SCENARIOS[scenario]),
                anomaly, factor)
            score = L.localize_direct(art["error"], normalize=True)
            rep = L.localization_report(score, _true_mask(art))
            rows.append({"scenario": scenario, "anomaly_type": anomaly,
                         "localization": rep["auroc_within"]})
    if not rows:
        print("  no p=128 artifacts found - skipping fig8")
        return None

    loc = pd.DataFrame(rows).groupby("scenario")["localization"].mean()
    det = (summary_by_seed[summary_by_seed["scenario"].isin(scenarios)]
           .groupby("scenario")["AUROC_mean"].mean())
    order = [s for s in scenarios if s in loc.index and s in det.index]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - 0.2, det[order].values, width=0.38, color="#4575b4",
           label="detection  (window-level AUROC)")
    ax.bar(x + 0.2, loc[order].values, width=0.38, color="#1b7837",
           label="localization  (within-window AUROC)")
    ax.axhline(0.5, color="black", lw=1.3, ls="--")
    ax.text(0.99, 0.52, "chance = 0.50", fontsize=11, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s].replace(" ", "\n") for s in order])
    ax.set_ylabel("AUROC, averaged over the four anomaly types")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Broken training data hurts detection, not attribution")
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_leakage_by_anomaly(out_path, toy_root, base_dir, scenario="toy_chain",
                           seed=2, context=40):
    """Why the causal term can only ever help for *some* anomaly types.

    Propagation gathers evidence from a culprit's children, so it needs the
    disturbance to actually reach them. Measuring that (``leakage_report``) shows
    the two regimes plainly: a single-timestep spike leaves the children almost
    untouched, while a collective anomaly drags them along by more than an order
    of magnitude. Hence the term is dead weight for point anomalies - and, at the
    leaf, worse than dead weight.
    """
    cases = [("point_global", "3.0", "point anomaly (single spike)"),
             ("collective_trend", "3.0", "collective anomaly (drift)")]
    loaded = []
    for anomaly, factor, label in cases:
        art = _load_toy(toy_root, scenario, seed, anomaly, factor, base_dir)
        if art is None:
            print("  missing toy artifacts - skipping fig9")
            return None
        rep = L.leakage_report(L.localize_direct(art["error"]),
                              art["true_mask"], art["graph"])
        loaded.append((art, rep, label))

    n_chain = min(3, loaded[0][0]["error"].shape[1])
    role = {0: "root", 1: "middle", 2: "leaf"}
    fig, axes = plt.subplots(n_chain, 2, figsize=(12.5, 6.6), sharex="col")

    for col, (art, rep, label) in enumerate(loaded):
        error, true_mask = art["error"], art["true_mask"]
        culprit = int(np.argmax(true_mask.any(axis=0)))
        hits = np.where(true_mask[:, culprit])[0]
        t = int(hits[len(hits) // 2])
        lo, hi = max(0, t - context), min(len(error), t + context)
        for v in range(n_chain):
            ax = axes[v, col]
            colour = POSITION_COLORS.get(
                {i: s for s, i in TOY_SCENARIOS.items()}.get(v), "#777777")
            ax.plot(range(lo, hi), error[lo:hi, v], color=colour, lw=1.7)
            ax.axvline(t, color="#d73027", lw=1.3, ls="--")
            ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(f"var {v}\n({role.get(v, '')})")
            if v == 0:
                ax.set_title(f"{label}\nchild/unrelated error ratio = "
                             f"{rep['children'] / rep['other']:.1f}x")
        axes[-1, col].set_xlabel("window index")
        axes[-1, col].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

    fig.suptitle("Propagation needs the disturbance to reach the children - "
                 "point anomalies stay put", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build the presentation figures.")
    p.add_argument("--results-root", default="results",
                   help="Folder holding the aggregated CSVs.")
    p.add_argument("--out-dir", default="results/slides",
                   help="Where to write the figures (default: results/slides).")
    p.add_argument("--toy-root", default=None,
                   help="Results tree with the toy .npy artifacts (enables fig5).")
    p.add_argument("--step1-root", default=None,
                   help="Results tree with the p=128 .npy artifacts (enables fig7).")
    p.add_argument("--base-dir", default="data",
                   help="Dataset folder, for the per-variable ground truth.")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    _style()

    def _read(name):
        path = os.path.join(args.results_root, name)
        if not os.path.exists(path):
            print(f"  missing {path} - skipping the figures that need it")
            return None
        return pd.read_csv(path)

    summary = _read("summary_by_seed.csv")
    metrics = _read("localization_metrics.csv")
    sweep = _read("localization_alpha_sweep.csv")

    if summary is not None:
        print("wrote", fig_detection_robustness(
            summary, os.path.join(out_dir, "fig1_detection_robustness.png")))
    if metrics is not None:
        print("wrote", fig_localization_position(
            metrics, os.path.join(out_dir, "fig2_localization_position.png")))
        for sc, tag in (("toy_chain", ""), ("toy_chain_mid", "_mid")):
            out = fig_simple_localization_hit(
                metrics, os.path.join(out_dir, f"fig2b_simple_hit{tag}.png"),
                scenario=sc)
            if out:
                print("wrote", out)
    if sweep is not None:
        print("wrote", fig_propagation_alpha(
            sweep, os.path.join(out_dir, "fig3_propagation_alpha.png")))

    out = fig_toy_system(os.path.join(out_dir, "fig4_toy_system.png"),
                         base_dir=args.base_dir)
    if out:
        print("wrote", out)
    if args.toy_root:
        for sc, tag in (("toy_chain", ""), ("toy_chain_mid", "_mid")):
            out = fig_simple_example_window(
                os.path.join(out_dir, f"fig2a_simple_example{tag}.png"),
                toy_root=args.toy_root, base_dir=args.base_dir, scenario=sc)
            if out:
                print("wrote", out)
            out = fig_simple_examples_grid(
                os.path.join(out_dir, f"fig2c_simple_examples_grid{tag}.png"),
                toy_root=args.toy_root, base_dir=args.base_dir, scenario=sc)
            if out:
                print("wrote", out)
        out = fig_propagation_mechanism(
            os.path.join(out_dir, "fig5_propagation_mechanism.png"),
            toy_root=args.toy_root, base_dir=args.base_dir)
        if out:
            print("wrote", out)
    if sweep is not None:
        out = fig_propagation_both(
            sweep, os.path.join(out_dir, "fig6_propagation_both.png"))
        if out:
            print("wrote", out)
    if args.step1_root:
        out = fig_highdim_paradox(
            os.path.join(out_dir, "fig7_highdim_paradox.png"),
            step1_root=args.step1_root, base_dir=args.base_dir)
        if out:
            print("wrote", out)
        if summary is not None:
            out = fig_detection_vs_localization(
                os.path.join(out_dir, "fig8_detection_vs_localization.png"),
                summary, step1_root=args.step1_root, base_dir=args.base_dir)
            if out:
                print("wrote", out)

    if args.toy_root:
        out = fig_leakage_by_anomaly(
            os.path.join(out_dir, "fig9_leakage_by_anomaly.png"),
            toy_root=args.toy_root, base_dir=args.base_dir)
        if out:
            print("wrote", out)


if __name__ == "__main__":
    main()
