# Nonstationary Data Generation – Changes (A + C)

Date: 2026-07-24  
File: `experiments/datagen.py`  
Related finding: Finding 2 (`nonstationary` ≈ `baseline` in AUROC)

## Problem

The previous `simulate_nonstationary` implementation only switched the
**causal graph** across regimes. Within each regime:

- `auto_corr` and `sd` were **globally fixed** (3.0 and 0.1),
- `make_var_stationary` made each regime **stable on its own**.

The **marginal** distribution of each variable therefore stayed almost like
`baseline` → CAROTS showed essentially no performance drop.

## What changed

### Option A – Per-regime marginal dynamics

Each regime now independently samples:

| Parameter   | Old value | New (default range) |
|-------------|-----------|---------------------|
| `auto_corr` | fixed 3.0 | Uniform `[1.5, 4.0]` |
| `sd`        | fixed 0.1 | Uniform `[0.05, 0.30]` |

The causal graph still changes as before. In addition, **persistence** and
**noise scale** jump at regime boundaries.

CLI overrides:

```bash
python -m experiments.datagen --variant nonstationary \
  --auto-corr-range 1.5 4.0 --sd-range 0.05 0.30
```

### Option C – More regimes

| Parameter   | Old | New (default) |
|-------------|-----|---------------|
| `n_regimes` | 4   | **8**         |

More switches inside the train and test splits → violations of the
"time-invariant causality" assumption are more frequent.

```bash
python -m experiments.datagen --variant nonstationary --n-regimes 8
```

### Helper `_simulate_from_betas`

`sd` may now be a **scalar or a sequence** (one `sd` per regime).
Noise is drawn as standard normal and scaled by the active regime's `sd`.

### New artifacts under `data/VAR_nonstationary/`

| File                      | Content |
|---------------------------|---------|
| `GC_regimes.npy`          | Graph per regime `(n_regimes, N, N)` – same idea as before |
| `regime_params.npz`       | **new:** `auto_corr`, `sd`, `bounds` per regime |
| `stationarity_check.txt`  | **new** (with `--check-adf`): rolling stats + ADF report |

## Verification with `--check-adf`

```bash
python -m experiments.datagen --variant nonstationary --check-adf
```

Compares the generated nonstationary series to a **same-seed baseline VAR**:

1. **Rolling mean / std** (primary): relative change first vs second half.
   If A worked, `std |Δ|` (and often `mean |Δ|`) on the nonstationary data
   should be **clearly larger** than on the baseline.
2. **ADF** (complementary): unit-root test on a sample of variables.
   Requires `statsmodels` (`pip install statsmodels`).

### ADF short note

ADF tests **unit-root** nonstationarity (e.g. random walk), **not**
automatically variance switches across regimes. Stable VARs (even with
regime-wise `sd` jumps) can still look ADF-"stationary". Rolling statistics
are therefore the decisive check for Option A.

## What you need to do next

1. Regenerate the data (overwrite old `data/VAR_nonstationary/`):

   ```bash
   python -m experiments.datagen --variant nonstationary --check-adf
   ```

2. Read `stationarity_check.txt` – rolling `std |Δ|` should clearly exceed the baseline.

3. Retrain the `nonstationary` experiments (same seeds as before), then run
   `aggregate` – only then can Finding 2 be re-evaluated empirically.

`baseline` / `nocausal` / `contaminated` are **not** affected by A+C
(unless you intentionally regenerate them with `--variant all`).

## Unchanged

- Anomaly injection (`factor`, `ratio`, types)
- Train/test split 50/50
- Filenames expected by `VARSegLoader` (`train.npy`, `test_*_outliers_…`)
- Threshold / metric pipeline (oracle-threshold finding) – separate topic
