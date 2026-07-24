"""Generate VAR datasets for the Step 1 robustness study.

We reuse the upstream VAR simulator (``data/VAR/simu_data.py``) and the anomaly
injector (``data/VAR/multivariate_generator.py``) so that our "clean baseline"
is identical to the dataset used in the CAROTS paper. On top of that we add
three *flawed training data* variants:

  baseline       : the standard VAR process from the paper (reference point).
  nocausal       : a VAR process with NO cross-variable causal links - every
                   variable is an independent auto-regressive series. The
                   ground-truth causal graph is the identity matrix.
  nonstationary  : a VAR process whose causal graph AND marginal dynamics
                   CHANGE over time (several regimes). This violates CAROTS'
                   assumption that causal relationships are time-invariant.
                   See ``CHANGELOG – nonstationary (A+C)`` below.
  contaminated   : the clean baseline process, but with synthetic anomalies
                   injected into the TRAINING split. This violates the
                   unsupervised assumption that training data is anomaly-free.
                   Its test set is identical to the baseline test set, so the
                   only thing that changes is the quality of the training data.

Every variant is written into its own directory using the SAME file layout that
``VARSegLoader`` expects, plus one extra file per test set:

  train.npy                                  (T_train, N)
  GC.npy                                     (N, N)   ground-truth causal graph
  test_<type>_outliers_factor<f>.npy         (T_test, N)
  test_<type>_outliers_factor<f>_labels.npy  (T_test,)   per-timestep labels
  test_<type>_outliers_factor<f>_varlabels.npy (T_test, N) per-VARIABLE labels
                                             (added by us, used for the Step 2
                                             localization visualizations)

For ``nonstationary`` we additionally write:

  GC_regimes.npy           (n_regimes, N, N)  graph per regime
  regime_params.npz        auto_corr / sd / bounds per regime (reproducibility)
  stationarity_check.txt   ADF + rolling mean/std diagnostics (if --check-adf)

Run examples (from the CAROTS repo root, with the project venv active):

  python -m experiments.datagen --variant baseline      --out-dir data/VAR_baseline
  python -m experiments.datagen --variant nocausal      --out-dir data/VAR_nocausal
  python -m experiments.datagen --variant nonstationary --out-dir data/VAR_nonstationary --check-adf
  python -m experiments.datagen --variant contaminated  --out-dir data/VAR_contaminated

  # or generate all four at once into data/VAR_<variant>/:
  python -m experiments.datagen --variant all --check-adf

---------------------------------------------------------------------------
CHANGELOG – nonstationary strengthening (A + C), 2026-07
---------------------------------------------------------------------------
Problem (Finding 2): The previous ``simulate_nonstationary`` only switched the
*causal graph* across regimes while keeping ``auto_corr`` and ``sd`` identical
and calling ``make_var_stationary`` per regime. Each regime was marginally
almost like the baseline → CAROTS saw ~no AUROC gap vs baseline.

Changes (Option A + C):
  A) Per-regime *marginal* dynamics: sample ``auto_corr`` and ``sd`` independently
     for every regime from configurable ranges (defaults:
     auto_corr ∈ [1.5, 4.0], sd ∈ [0.05, 0.30]). The causal graph still changes
     as before; now variance / persistence also jump at regime boundaries.
  C) More regimes by default: ``n_regimes`` default raised from 4 → 8 so the
     process switches more often within train and test splits.

Verification:
  ``verify_nonstationarity`` (CLI ``--check-adf``) compares a series against a
  same-seed *baseline* VAR on:
    - rolling mean / std (most relevant for A: variance/level shifts)
    - Augmented Dickey–Fuller (ADF) unit-root test on a sample of variables
  Results are printed and written to ``stationarity_check.txt``.

NOTE on ADF: ADF tests *unit-root* nonstationarity (e.g. random walk), not
every form of nonstationarity. Regime-wise variance changes can still look
"ADF-stationary". Use rolling mean/std as the primary check that A worked;
treat ADF as complementary evidence.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# The upstream VAR helpers live in ``data/VAR`` and are written as loose scripts
# (not an importable package), so we add that folder to the import path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VAR_DIR = os.path.join(_REPO_ROOT, "data", "VAR")
if _VAR_DIR not in sys.path:
    sys.path.insert(0, _VAR_DIR)

from simu_data import make_var_stationary, simulate_var  # noqa: E402
from multivariate_generator import MultivariateDataGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Default parameters (match the paper's VAR setup unless noted).
# ---------------------------------------------------------------------------
DEFAULT_P = 128            # number of variables
DEFAULT_LENGTH = 40000     # total length; split 50/50 into train/test
DEFAULT_LAG = 3
DEFAULT_SPARSITY = 0.2     # fraction of off-diagonal causal links (baseline)
DEFAULT_BETA_VALUE = 1.0
DEFAULT_AUTO_CORR = 3.0
DEFAULT_SD = 0.1

# Nonstationary defaults (A + C). Previously n_regimes=4 with fixed auto_corr/sd.
DEFAULT_N_REGIMES = 8
DEFAULT_AUTO_CORR_RANGE = (1.5, 4.0)   # A: persistence varies across regimes
DEFAULT_SD_RANGE = (0.05, 0.30)        # A: noise scale varies across regimes

# Anomaly injection (test set) - same settings as the upstream generate.py.
ANOMALY_VAR_NUM = 10       # how many variables receive anomalies
ANOMALY_RATIO = 0.01
ANOMALY_RADIUS = 5

# Training-set contamination (the "contaminated" variant only).
DEFAULT_CONTAMINATION_RATIO = 0.05  # ~5% of training timesteps become anomalous


# ---------------------------------------------------------------------------
# VAR coefficient / data simulation helpers.
# ---------------------------------------------------------------------------
def _build_var_beta(p, lag, sparsity, beta_value, auto_corr, rng):
    """Build a VAR coefficient matrix and its Granger-causality graph.

    This mirrors the upstream ``simulate_var`` coefficient construction but uses
    an explicit RNG and guards the ``sparsity == 0`` case so we can build a
    causal-link-free process.

    Returns ``(beta, GC)`` where ``beta`` has shape ``(p, p*lag)`` and ``GC`` is
    the ``(p, p)`` 0/1 ground-truth causal graph.

    ``make_var_stationary`` rescales eigenvalues so the *within-regime* VAR is
    stable (no exploding trajectories). That is orthogonal to *across-regime*
    nonstationarity (changing graph / sd / auto_corr over time).
    """
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * auto_corr

    # Upstream uses ``int(p * sparsity) - 1`` which becomes -1 (a crash) when
    # sparsity is 0. We clamp at 0 so "no causal links" is well defined.
    num_nonzero = max(0, int(p * sparsity) - 1)
    for i in range(p):
        if num_nonzero == 0:
            continue
        choice = rng.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)
    return beta, GC


def _simulate_from_betas(betas, regime_bounds, p, lag, T, sd, rng):
    """Simulate a VAR series, switching the coefficient matrix per regime.

    ``betas`` is a list of ``(p, p*lag)`` matrices and ``regime_bounds`` is a
    list of the same length giving the *exclusive* end index (in the post
    burn-in timeline) of each regime. With a single beta this reduces to the
    standard stationary VAR simulation.

    ``sd`` may be a scalar (same noise everywhere) or a sequence of length
    ``len(betas)`` (per-regime noise scale – used by the strengthened
    nonstationary simulator).
    """
    burn_in = 100
    total = T + burn_in
    # Draw unit-variance noise; scale by the active regime's sd at each step.
    errors = rng.normal(loc=0, scale=1.0, size=(p, total))
    X = np.ones((p, total))
    X[:, :lag] = errors[:, :lag]

    if np.isscalar(sd):
        sds = [float(sd)] * len(betas)
    else:
        sds = [float(s) for s in sd]
        if len(sds) != len(betas):
            raise ValueError("sd sequence length must match number of regimes")

    for t in range(lag, total):
        post_idx = max(0, t - burn_in)
        # find the regime this timestep belongs to
        regime = 0
        for r, bound in enumerate(regime_bounds):
            if post_idx < bound:
                regime = r
                break
        else:
            regime = len(betas) - 1
        beta = betas[regime]
        X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order="F"))
        X[:, t] += errors[:, t - 1] * sds[regime]

    return X.T[burn_in:, :]


def simulate_nocausal(p, T, lag, beta_value, auto_corr, sd, seed):
    """VAR with only auto-correlation: no cross-variable causal links."""
    rng = np.random.RandomState(seed)
    beta, GC = _build_var_beta(p, lag, sparsity=0.0, beta_value=beta_value,
                               auto_corr=auto_corr, rng=rng)
    data = _simulate_from_betas([beta], [T], p, lag, T, sd, rng)
    return data, GC


def simulate_nonstationary(
        p, T, lag, sparsity, beta_value, auto_corr, sd, seed,
        n_regimes=DEFAULT_N_REGIMES,
        auto_corr_range: Tuple[float, float] = DEFAULT_AUTO_CORR_RANGE,
        sd_range: Tuple[float, float] = DEFAULT_SD_RANGE,
):
    """VAR with time-varying causal graph AND marginal dynamics (A + C).

    Parameters
    ----------
    auto_corr, sd :
        Kept for API compatibility with the old call site. They are *ignored*
        when ``auto_corr_range`` / ``sd_range`` are used (the default). Pass
        identical min=max ranges to recover a fixed value.
    n_regimes :
        Number of piecewise regimes along the timeline (C: default 8, was 4).
    auto_corr_range, sd_range :
        Uniform sampling ranges for per-regime persistence and noise (A).

    Returns
    -------
    data : (T, N) array
    gc_union : (N, N) OR of all regime graphs (for GC.npy)
    gc_regimes : (n_regimes, N, N)
    regime_params : dict with keys auto_corr, sd, bounds (for logging / npz)
    """
    rng = np.random.RandomState(seed)
    betas, gcs = [], []
    regime_auto_corr, regime_sd = [], []

    for _ in range(n_regimes):
        # A: draw regime-specific marginal parameters
        ac = float(rng.uniform(auto_corr_range[0], auto_corr_range[1]))
        sd_r = float(rng.uniform(sd_range[0], sd_range[1]))
        beta, gc = _build_var_beta(p, lag, sparsity, beta_value, ac, rng)
        betas.append(beta)
        gcs.append(gc)
        regime_auto_corr.append(ac)
        regime_sd.append(sd_r)

    # Equal-length regimes; last regime absorbs the remainder.
    seg = T // n_regimes
    regime_bounds = [(r + 1) * seg for r in range(n_regimes)]
    regime_bounds[-1] = T

    data = _simulate_from_betas(betas, regime_bounds, p, lag, T, regime_sd, rng)

    gc_union = np.clip(np.sum(np.stack(gcs, axis=0), axis=0), 0, 1).astype(int)
    regime_params = {
        "auto_corr": np.asarray(regime_auto_corr),
        "sd": np.asarray(regime_sd),
        "bounds": np.asarray(regime_bounds),
        "auto_corr_range": np.asarray(auto_corr_range),
        "sd_range": np.asarray(sd_range),
        # store unused legacy args so the npz is self-describing
        "legacy_auto_corr_arg": np.asarray(auto_corr),
        "legacy_sd_arg": np.asarray(sd),
    }
    print(f"  nonstationary regimes={n_regimes}")
    print(f"    auto_corr per regime: "
          f"{np.array2string(regime_params['auto_corr'], precision=3)}")
    print(f"    sd        per regime: "
          f"{np.array2string(regime_params['sd'], precision=3)}")
    return data, gc_union, np.stack(gcs, axis=0), regime_params


# ---------------------------------------------------------------------------
# Stationarity diagnostics (ADF + rolling mean/std)
# ---------------------------------------------------------------------------
def _rolling_mean_std(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple rolling mean / std for a 1-D series (valid from index window-1)."""
    if window < 2 or window > len(series):
        raise ValueError(f"invalid rolling window={window} for length={len(series)}")
    # cumsum trick for O(T) rolling stats
    c1 = np.cumsum(np.insert(series, 0, 0.0))
    c2 = np.cumsum(np.insert(series ** 2, 0, 0.0))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    mean = s1 / window
    var = np.maximum(s2 / window - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def _adf_pvalue(series: np.ndarray) -> Optional[float]:
    """Return ADF p-value, or None if statsmodels is not installed."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
    # regression='c': constant, no trend — standard choice for this check
    result = adfuller(series, maxlag=None, regression="c", autolag="AIC")
    return float(result[1])  # p-value


def verify_nonstationarity(
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        n_vars_sample: int = 16,
        rolling_window: Optional[int] = None,
        seed: int = 0,
        out_path: Optional[str] = None,
) -> str:
    """Compare ``data`` (e.g. nonstationary) to an optional stationary reference.

    Reports for a random sample of variables:
      * relative change of rolling mean / std between first and second half
        (primary evidence that Option A introduced marginal shifts)
      * ADF p-values (complementary; see module docstring note on ADF)

    Returns the report text (also printed; optionally written to ``out_path``).
    """
    rng = np.random.RandomState(seed)
    T, N = data.shape
    if rolling_window is None:
        rolling_window = max(100, T // 20)
    n_vars_sample = min(n_vars_sample, N)
    var_ids = np.sort(rng.choice(N, size=n_vars_sample, replace=False))

    lines: List[str] = []
    lines.append("=== Stationarity / nonstationarity check ===")
    lines.append(f"series shape: {data.shape}, sampled vars: {list(var_ids)}")
    lines.append(f"rolling window: {rolling_window}")
    lines.append("")

    def _half_shift(series: np.ndarray) -> Tuple[float, float]:
        mean, std = _rolling_mean_std(series, rolling_window)
        mid = len(mean) // 2
        # relative change between early and late rolling stats
        mean_rel = abs(mean[mid:].mean() - mean[:mid].mean()) / (
            abs(mean[:mid].mean()) + 1e-8)
        std_rel = abs(std[mid:].mean() - std[:mid].mean()) / (
            std[:mid].mean() + 1e-8)
        return float(mean_rel), float(std_rel)

    mean_rels, std_rels, adf_ps = [], [], []
    for v in var_ids:
        mr, sr = _half_shift(data[:, v])
        mean_rels.append(mr)
        std_rels.append(sr)
        p = _adf_pvalue(data[:, v])
        adf_ps.append(p)

    lines.append("[target series] rolling half-vs-half relative change "
                 f"(mean over {n_vars_sample} vars):")
    lines.append(f"  mean |Δ|: {np.mean(mean_rels):.4f}   "
                 f"std |Δ|: {np.mean(std_rels):.4f}")

    if all(p is None for p in adf_ps):
        lines.append("  ADF: statsmodels not installed "
                     "(pip install statsmodels). Skipping unit-root test.")
    else:
        # None → treat as missing
        valid = [p for p in adf_ps if p is not None]
        n_reject = sum(p < 0.05 for p in valid)  # reject H0 "has unit root"
        lines.append(f"  ADF: {n_reject}/{len(valid)} vars reject unit-root "
                     f"at α=0.05 (median p={np.median(valid):.4g})")
        lines.append("       (rejecting unit root ≠ 'no regime shifts'; "
                     "see module docstring)")

    if reference is not None:
        assert reference.shape == data.shape
        r_mean, r_std, r_adf = [], [], []
        for v in var_ids:
            mr, sr = _half_shift(reference[:, v])
            r_mean.append(mr)
            r_std.append(sr)
            r_adf.append(_adf_pvalue(reference[:, v]))
        lines.append("")
        lines.append("[baseline reference] rolling half-vs-half relative change:")
        lines.append(f"  mean |Δ|: {np.mean(r_mean):.4f}   "
                     f"std |Δ|: {np.mean(r_std):.4f}")
        if all(p is None for p in r_adf):
            lines.append("  ADF: skipped (no statsmodels)")
        else:
            valid = [p for p in r_adf if p is not None]
            n_reject = sum(p < 0.05 for p in valid)
            lines.append(f"  ADF: {n_reject}/{len(valid)} vars reject unit-root "
                         f"at α=0.05 (median p={np.median(valid):.4g})")
        lines.append("")
        lines.append("Interpretation guide:")
        lines.append("  • Larger rolling std |Δ| on the target vs baseline ⇒")
        lines.append("    Option A (per-regime sd / auto_corr) is visible in data.")
        lines.append("  • ADF rejecting unit roots on both series is normal for")
        lines.append("    stable VARs; do NOT expect ADF alone to 'prove' A+C.")

    report = "\n".join(lines) + "\n"
    print(report)
    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  wrote {out_path}")
    return report


# ---------------------------------------------------------------------------
# Anomaly injection (test set) + training-set contamination.
# ---------------------------------------------------------------------------
def _inject_and_save(test, out_dir, factors, seed):
    """Inject the four synthetic anomaly types into the test set and save.

    For every saved test file we also save a per-variable label matrix
    (shape ``(T_test, N)``) recovered by diffing the injected series against the
    untouched original - this is the ground truth used by the Step 2
    localization visualizations.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _save_one(method, factor, radius):
        # A fresh generator per call keeps anomalies independent across files and
        # mirrors the upstream behaviour (one generator instance, fresh draws).
        np.random.seed(seed)
        gen = MultivariateDataGenerator(test.transpose())  # expects (N, T)
        func = getattr(gen, method)
        if factor is None:
            out, lab = func(var_num=ANOMALY_VAR_NUM, ratio=ANOMALY_RATIO, radius=radius)
        else:
            out, lab = func(var_num=ANOMALY_VAR_NUM, ratio=ANOMALY_RATIO,
                            factor=factor, radius=radius)

        out_t = out.transpose()                       # (T, N)
        var_lab = (out != gen.data_origin).astype(int).transpose()  # (T, N)

        base = f"test_{method}_factor{factor}"
        np.save(os.path.join(out_dir, base + ".npy"), out_t)
        np.save(os.path.join(out_dir, base + "_labels.npy"), lab)
        np.save(os.path.join(out_dir, base + "_varlabels.npy"), var_lab)
        n_anom_vars = int((var_lab.sum(axis=0) > 0).sum())
        print(f"  saved {base}.npy  (anomalous vars: {n_anom_vars}, "
              f"anomalous steps: {int(lab.sum())})")

    for factor in factors:
        _save_one("point_global_outliers", factor=factor, radius=ANOMALY_RADIUS)
        _save_one("point_contextual_outliers", factor=factor, radius=ANOMALY_RADIUS)
        _save_one("collective_trend_outliers", factor=factor, radius=ANOMALY_RADIUS)
    # collective_global does not use a factor
    _save_one("collective_global_outliers", factor=None, radius=ANOMALY_RADIUS)


def _contaminate_train(train, ratio, seed):
    """Inject anomalies into the TRAINING split (for the contaminated variant).

    We mix point-global and collective-global anomalies so the contamination
    contains both spike-like and pattern-like anomalies. Returns the
    contaminated training array and a 0/1 contamination mask of shape
    ``(T_train,)``.
    """
    np.random.seed(seed + 777)
    gen = MultivariateDataGenerator(train.transpose())  # (N, T)

    # Split the contamination budget across two anomaly types.
    half = ratio / 2.0
    out_pg, lab_pg = gen.point_global_outliers(
        var_num=ANOMALY_VAR_NUM, ratio=half, factor=3.0, radius=ANOMALY_RADIUS)
    gen.data = out_pg  # chain the second anomaly type on top of the first
    out_cg, lab_cg = gen.collective_global_outliers(
        var_num=ANOMALY_VAR_NUM, ratio=half, radius=ANOMALY_RADIUS)

    contaminated = out_cg.transpose()                  # (T, N)
    mask = np.clip(lab_pg + lab_cg, 0, 1).astype(int)  # (T,)
    print(f"  contaminated {int(mask.sum())} / {len(mask)} training steps "
          f"({100.0 * mask.mean():.2f}%)")
    return contaminated, mask


# ---------------------------------------------------------------------------
# Top-level variant generation.
# ---------------------------------------------------------------------------
def generate_variant(variant, out_dir, factors, seed,
                     length=DEFAULT_LENGTH, p=DEFAULT_P, lag=DEFAULT_LAG,
                     contamination_ratio=DEFAULT_CONTAMINATION_RATIO,
                     n_regimes=DEFAULT_N_REGIMES,
                     auto_corr_range=DEFAULT_AUTO_CORR_RANGE,
                     sd_range=DEFAULT_SD_RANGE,
                     check_adf: bool = False):
    """Generate one dataset variant into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    half = length // 2
    print(f"[{variant}] generating into {out_dir} (seed={seed}) ...")

    if variant == "baseline":
        data, _beta, GC = simulate_var(p=p, T=length, lag=lag, seed=seed)
        train, test = data[:half], data[half:]

    elif variant == "nocausal":
        data, GC = simulate_nocausal(p=p, T=length, lag=lag,
                                     beta_value=DEFAULT_BETA_VALUE,
                                     auto_corr=DEFAULT_AUTO_CORR,
                                     sd=DEFAULT_SD, seed=seed)
        train, test = data[:half], data[half:]

    elif variant == "nonstationary":
        data, GC, gc_regimes, regime_params = simulate_nonstationary(
            p=p, T=length, lag=lag, sparsity=DEFAULT_SPARSITY,
            beta_value=DEFAULT_BETA_VALUE, auto_corr=DEFAULT_AUTO_CORR,
            sd=DEFAULT_SD, seed=seed, n_regimes=n_regimes,
            auto_corr_range=auto_corr_range, sd_range=sd_range)
        train, test = data[:half], data[half:]
        np.save(os.path.join(out_dir, "GC_regimes.npy"), gc_regimes)
        np.savez(os.path.join(out_dir, "regime_params.npz"), **regime_params)
        print(f"  wrote GC_regimes.npy and regime_params.npz")

        if check_adf:
            # Same-seed stationary baseline as reference for the diagnostic.
            ref, _beta, _gc = simulate_var(p=p, T=length, lag=lag, seed=seed)
            verify_nonstationarity(
                data, reference=ref, seed=seed,
                out_path=os.path.join(out_dir, "stationarity_check.txt"),
            )

    elif variant == "contaminated":
        # Same clean process as the baseline; only the training split changes.
        data, _beta, GC = simulate_var(p=p, T=length, lag=lag, seed=seed)
        train_clean, test = data[:half], data[half:]
        train, mask = _contaminate_train(train_clean, contamination_ratio, seed)
        np.save(os.path.join(out_dir, "train_contamination_mask.npy"), mask)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    np.save(os.path.join(out_dir, "train.npy"), train)
    np.save(os.path.join(out_dir, "GC.npy"), GC)
    print(f"  train shape {train.shape}, test shape {test.shape}")

    _inject_and_save(test, out_dir, factors=factors, seed=seed)
    print(f"[{variant}] done.\n")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "nocausal", "nonstationary",
                                 "contaminated", "all"],
                        help="which dataset variant to generate")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: data/VAR_<variant>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factors", type=float, nargs="+", default=[2.0],
                        help="anomaly difficulty factors for the test set "
                             "(default: 2.0, the paper default)")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--p", type=int, default=DEFAULT_P,
                        help="number of variables")
    parser.add_argument("--contamination-ratio", type=float,
                        default=DEFAULT_CONTAMINATION_RATIO)
    parser.add_argument("--n-regimes", type=int, default=DEFAULT_N_REGIMES,
                        help="number of regimes for nonstationary "
                             f"(default {DEFAULT_N_REGIMES}; was 4 before A+C)")
    parser.add_argument("--auto-corr-range", type=float, nargs=2,
                        default=list(DEFAULT_AUTO_CORR_RANGE),
                        metavar=("LOW", "HIGH"),
                        help="per-regime auto_corr sampling range (Option A)")
    parser.add_argument("--sd-range", type=float, nargs=2,
                        default=list(DEFAULT_SD_RANGE),
                        metavar=("LOW", "HIGH"),
                        help="per-regime noise-sd sampling range (Option A)")
    parser.add_argument("--check-adf", action="store_true",
                        help="for nonstationary: write stationarity_check.txt "
                             "(rolling stats + ADF vs same-seed baseline)")
    return parser.parse_args()


def main():
    args = _parse_args()
    variants = (["baseline", "nocausal", "nonstationary", "contaminated"]
                if args.variant == "all" else [args.variant])
    for variant in variants:
        if len(variants) > 1:
            # Generating several variants: treat --out-dir (if given) as a parent
            # folder so each variant lands in its own subdirectory.
            parent = args.out_dir or os.path.join(_REPO_ROOT, "data")
            out_dir = os.path.join(parent, f"VAR_{variant}")
        else:
            out_dir = args.out_dir or os.path.join(_REPO_ROOT, "data", f"VAR_{variant}")
        generate_variant(
            variant=variant,
            out_dir=out_dir,
            factors=args.factors,
            seed=args.seed,
            length=args.length,
            p=args.p,
            contamination_ratio=args.contamination_ratio,
            n_regimes=args.n_regimes,
            auto_corr_range=tuple(args.auto_corr_range),
            sd_range=tuple(args.sd_range),
            check_adf=args.check_adf,
        )


if __name__ == "__main__":
    main()
