# Toy causal chain (Step 2 localization demo)

Date: 2026-07-24  
Related code: `experiments/datagen.py` (`--variant toy_chain`),
`experiments/scenarios.py` (`toy_chain` scenario)

## Why this exists

On the full VAR setup (`p=128`, ~10 anomalous variables) the localization
heatmaps are hard to read: many variables light up, point anomalies are rare,
and it is unclear whether the model found the *root cause*.

`toy_chain` is a **minimal** synthetic system with:

1. A **clear linear causal chain** `0 → 1 → 2`
2. A few **independent distractor** variables (default: 2 → total `p=5`)
3. Test anomalies injected into the **root only** (variable `0`)

So the correct localization answer is always: **variable 0**.

## Generative model

For lag 1 and coupling `couple` (default `0.85`):

```
X0_t = a * X0_{t-1} + eps
X1_t = couple * X0_{t-1} + a * X1_{t-1} + eps
X2_t = couple * X1_{t-1} + a * X2_{t-1} + eps
```

Distractors are independent AR(1). Ground-truth `GC.npy` uses the same
convention as `localization.py`: `GC[i, j] == 1` means **i causes j**.

Anomaly files use factor **3.0** by default (stronger than the paper's 2.0).

## Generate data

```bash
python -m experiments.datagen --variant toy_chain
# → data/VAR_toy_chain/{train.npy, GC.npy, toy_meta.npz, test_*…}
```

Useful knobs:

```bash
python -m experiments.datagen --variant toy_chain \
  --toy-chain-len 3 --toy-distractors 2 --toy-couple 0.85 --toy-root-var 0
```

`--variant all` does **not** include `toy_chain` (Step-1 grid stays unchanged).

## Train + localize

`utils/parser.py` reads `N_VAR` from `data/VAR_toy_chain/train.npy`, so no
manual `DATA.N_VAR` override is required.

```bash
python -m experiments.run_experiments --scenarios toy_chain --seeds 2
bash experiments/generated/run_toy_chain.sh
python -m experiments.aggregate
```

Figures land under `results/localization/toy_chain/seed2/`.

What to look for:

| Plot | Expectation |
|------|-------------|
| Bottom heatmap panel | Only variable **0** marked anomalous |
| Top (`direct`) | Root bright; children may also light up (error leaks downstream) |
| Top (`causal`) | Credit pushed toward the root via the chain |
| Window bar chart | Variable **0** among the top ranks (ideally #1), marked **red** |

## Files written

| File | Content |
|------|---------|
| `train.npy` | `(T/2, 5)` clean training series |
| `GC.npy` | `(5, 5)` ground-truth graph |
| `toy_meta.npz` | chain length, couple, root_var, … |
| `test_*_varlabels.npy` | only column 0 is non-zero on anomalous steps |
