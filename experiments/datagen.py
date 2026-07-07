"""Generate VAR datasets for the Step 1 robustness study.

We reuse the upstream VAR simulator (``data/VAR/simu_data.py``) and the anomaly
injector (``data/VAR/multivariate_generator.py``) so that our "clean baseline"
is identical to the dataset used in the CAROTS paper. On top of that we add
three *flawed training data* variants:

  baseline       : the standard VAR process from the paper (reference point).
  nocausal       : a VAR process with NO cross-variable causal links - every
                   variable is an independent auto-regressive series. The
                   ground-truth causal graph is the identity matrix.
  nonstationary  : a VAR process whose causal graph CHANGES over time (it is
                   split into several "regimes", each with a different random
                   causal graph). This violates CAROTS' assumption that causal
                   relationships are time-invariant.
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

Run examples (from the CAROTS repo root, with the project venv active):

  python -m experiments.datagen --variant baseline      --out-dir data/VAR_baseline
  python -m experiments.datagen --variant nocausal      --out-dir data/VAR_nocausal
  python -m experiments.datagen --variant nonstationary --out-dir data/VAR_nonstationary
  python -m experiments.datagen --variant contaminated  --out-dir data/VAR_contaminated

  # or generate all four at once into data/VAR_<variant>/:
  python -m experiments.datagen --variant all
"""

from __future__ import annotations

import argparse
import os
import sys

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
    """
    burn_in = 100
    total = T + burn_in
    errors = rng.normal(loc=0, scale=sd, size=(p, total))
    X = np.ones((p, total))
    X[:, :lag] = errors[:, :lag]

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
        X[:, t] += errors[:, t - 1]

    return X.T[burn_in:, :]


def simulate_nocausal(p, T, lag, beta_value, auto_corr, sd, seed):
    """VAR with only auto-correlation: no cross-variable causal links."""
    rng = np.random.RandomState(seed)
    beta, GC = _build_var_beta(p, lag, sparsity=0.0, beta_value=beta_value,
                               auto_corr=auto_corr, rng=rng)
    data = _simulate_from_betas([beta], [T], p, lag, T, sd, rng)
    return data, GC


def simulate_nonstationary(p, T, lag, sparsity, beta_value, auto_corr, sd, seed,
                           n_regimes=4):
    """VAR whose causal graph changes over time (concept drift)."""
    rng = np.random.RandomState(seed)
    betas, gcs = [], []
    for _ in range(n_regimes):
        beta, gc = _build_var_beta(p, lag, sparsity, beta_value, auto_corr, rng)
        betas.append(beta)
        gcs.append(gc)

    seg = T // n_regimes
    regime_bounds = [(r + 1) * seg for r in range(n_regimes)]
    regime_bounds[-1] = T  # make sure the last regime covers the remainder

    data = _simulate_from_betas(betas, regime_bounds, p, lag, T, sd, rng)

    # Representative ground-truth graph = union of all regime graphs.
    gc_union = np.clip(np.sum(np.stack(gcs, axis=0), axis=0), 0, 1).astype(int)
    return data, gc_union, np.stack(gcs, axis=0)


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
                     n_regimes=4):
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
        data, GC, gc_regimes = simulate_nonstationary(
            p=p, T=length, lag=lag, sparsity=DEFAULT_SPARSITY,
            beta_value=DEFAULT_BETA_VALUE, auto_corr=DEFAULT_AUTO_CORR,
            sd=DEFAULT_SD, seed=seed, n_regimes=n_regimes)
        train, test = data[:half], data[half:]
        np.save(os.path.join(out_dir, "GC_regimes.npy"), gc_regimes)

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
    parser.add_argument("--n-regimes", type=int, default=4,
                        help="number of causal regimes for the nonstationary variant")
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
        )


if __name__ == "__main__":
    main()
