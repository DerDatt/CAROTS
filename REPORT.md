# Testing CAROTS under Flawed Training Data and Extending it with Variable-Level Anomaly Localization

> A reproducible study built on top of the official CAROTS implementation
> ([kimanki/CAROTS](https://github.com/kimanki/CAROTS), ICML 2025,
> *Causality-Aware Contrastive Learning for Robust Multivariate Time-Series
> Anomaly Detection*, arXiv:2506.03964).

This report documents two contributions:

- **Step 1 - Robustness study.** We stress-test CAROTS with three kinds of
  *flawed training data* and compare against a clean baseline, using fully
  controlled synthetic VAR datasets where the causal ground truth is known.
- **Step 2 - Variable-level anomaly localization.** We extend CAROTS so that,
  in addition to saying *when* an anomaly occurs, it says *which variable* is
  responsible - **without changing training, the model, or the loss functions**.

All code added for this study lives in the `CAROTS/experiments/` package plus a
handful of small, clearly-marked hooks in the upstream code. Training runs on a
CUDA GPU; data generation and analysis run on CPU. See
[`README.md`](README.md) for the one-command run instructions.

---

## 1. Background: how CAROTS works

CAROTS is an unsupervised multivariate time-series anomaly detector with three
stages:

1. **Causal discovery.** A forecasting-based causal discoverer (CUTS+) learns a
   causal graph `A` over the variables and learns to forecast each variable from
   its causal parents.
2. **Causality-aware contrastive learning.** Two augmentors generate views:
   a *causality-preserving augmentor* (adds noise to causes and re-derives
   effects through the graph) and a *causality-disturbing augmentor* (perturbs a
   sub-graph). An encoder is trained with a Similarity-filtered One-class
   Contrastive (SOC) loss so that causality-preserving views stay close to the
   data manifold while causality-disturbing views are pushed away.
3. **Anomaly scoring.** The final score combines two normalized terms:

   $$A(X) = z\big(A_\text{CL}\big) + z\big(A_\text{CD}\big)$$

   where $A_\text{CL}$ is the distance between the encoder embedding and the
   centroid of positive (normal) embeddings, and $A_\text{CD}$ is the
   mean-squared forecasting error of the causal discoverer. $z(\cdot)$ denotes
   z-normalization using training-set statistics.

**Key assumptions** (the ones we deliberately break in Step 1):

- *A1 - Causal structure exists and is informative.* The method is built around
  a meaningful causal graph among the variables.
- *A2 - Causal relationships are time-invariant (stationary).*
- *A3 - Training data is anomaly-free* (a standard unsupervised assumption).

---

## 2. Objectives

**Step 1.** Quantify how CAROTS degrades when each assumption is violated:

| Scenario        | Assumption broken | Description |
|-----------------|-------------------|-------------|
| `baseline`      | none              | Clean, stationary VAR with real causal links. |
| `nocausal`      | A1                | Variables are autocorrelation-only; the causal graph is the identity. |
| `nonstationary` | A2                | The causal graph drifts over time (multiple regimes). |
| `contaminated`  | A3                | Synthetic anomalies are injected into the **training** split. |

**Step 2.** Turn the *per-variable* forecasting error - which CAROTS already
computes internally and then averages away - into a variable-level anomaly
attribution, and propagate it along the known causal graph to emphasize
upstream root causes. This adds interpretability and is purely a post-processing
layer at inference time.

---

## 3. Data: a controlled VAR testbed

We use synthetic Vector Auto-Regressive (VAR) data because it gives us the
**ground-truth causal graph** and **ground-truth anomalous variables**, which is
exactly what is needed to evaluate causal robustness and localization.

The generator is [`CAROTS/experiments/datagen.py`](experiments/datagen.py).
It reuses the upstream VAR simulator (`data/VAR/simu_data.py`) and anomaly
injector (`data/VAR/multivariate_generator.py`) so the clean baseline matches
the paper's dataset, and adds the three flawed variants on top.

### 3.1 Common parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `p` (N variables) | 128 | dimensionality |
| length | 40000 | split 50/50 into train/test |
| lag | 3 | VAR order |
| sparsity | 0.2 | fraction of off-diagonal causal links (baseline) |
| auto_corr | 3.0 | diagonal self-coefficient |
| noise sd | 0.1 | innovation noise |

Anomalies are injected into the **test** split only (except for the
`contaminated` variant) using the four anomaly types from the paper, with
`var_num=10` variables affected, `ratio=0.01`, `radius=5`, and difficulty
`factor=2.0` (collective-global uses no factor):

- **Point Global (PG)** - isolated large spikes.
- **Point Contextual (PC)** - spikes that are only anomalous given context.
- **Collective Trend (CT)** - injected trends over a span.
- **Collective Global (CG)** - collective level shifts.

### 3.2 The four variants

- **`baseline`** - `simulate_var(p, T, lag)` exactly as upstream; GC is the real
  sparse graph.
- **`nocausal`** - the off-diagonal link budget is set to zero, so `beta` is
  diagonal and `GC = I`. Each variable is an independent AR(3) process. This
  removes the cross-variable causal signal CAROTS is designed to exploit.
- **`nonstationary`** - the timeline is split into `n_regimes=4` segments, each
  with its **own** randomly-drawn causal graph and coefficients. We save both a
  representative union graph (`GC.npy`) and the per-regime graphs
  (`GC_regimes.npy`). This breaks the time-invariance assumption (A2).
- **`contaminated`** - the process is the clean baseline, but ~5% of the
  **training** timesteps are corrupted with a mix of point-global and
  collective-global anomalies (`train_contamination_mask.npy` records which).
  Crucially, the **test set is identical to the baseline**, so any change in
  metrics isolates the effect of training contamination alone.

### 3.3 Per-variable ground truth (for Step 2)

For every test file we additionally save
`test_<type>_outliers_factor<f>_varlabels.npy` of shape `(T_test, N)`, recovered
by diffing the injected series against the untouched original. This per-variable
mask is the ground truth the localization figures are checked against.

### 3.4 Design rationale

- `nocausal` and `nonstationary` are **self-consistent**: train and test come
  from the same (flawed) process, with test anomalies injected on top. This asks
  *"can CAROTS still detect anomalies when the causal premise is wrong?"*
- `contaminated` deliberately **reuses the clean baseline test set** and only
  contaminates training, isolating the single factor of interest (A3).

---

## 4. Step 2: variable-level localization (design)

### 4.1 The signal already exists inside CAROTS

The causal-discrepancy score is the mean forecasting error over *time and
variables*:

$$A_\text{CD}(X_b) = \frac{1}{T\,N}\sum_{t,n}\big(\hat{y}_{b,t,n} - y_{b,t,n}\big)^2 .$$

If we stop averaging over the variable axis, we get a **per-variable error**:

$$E_{b,n} = \frac{1}{T}\sum_{t}\big(\hat{y}_{b,t,n} - y_{b,t,n}\big)^2 \in \mathbb{R}^{B\times N}.$$

A variable that suddenly becomes hard to forecast from its causal parents is a
natural anomaly candidate. This is implemented as
`Scorer.get_per_variable_cd_scores` in
[`models/carots/scorer_carots.py`](models/carots/scorer_carots.py); it
reuses the *exact* same forecast as the existing score and only changes which
axes are averaged. **No training, model, or loss change.**

### 4.2 Two attributions

Let $E$ be the per-variable error matrix (windows × variables) and let
$A$ be the binarized causal graph ($A_{ij}=1$ means variable $i$ causes $j$).

**(a) Direct.** Use $E$ directly, optionally robust-standardized per variable
(median / MAD) so each variable is judged against its own typical error level:

$$\text{direct}_{t,n} = \frac{E_{t,n} - \text{median}_n(E)}{1.4826\,\text{MAD}_n(E)} .$$

**(b) Causal-propagated.** Forecasting error *leaks downstream*: when a root-cause
variable is corrupted, its causal children inherit a corrupted input and also
look anomalous. To emphasize the upstream source, we add a fraction $\alpha$ of
the **mean error of each variable's causal children**:

$$\text{causal}_{t,i} = s_{t,i} + \alpha \cdot \frac{1}{|\text{children}(i)|}\sum_{j\in\text{children}(i)} s_{t,j} = s + \alpha\,(s\,\hat{A}^{\top}),$$

where $s$ is the (standardized) error and $\hat{A}$ is the row-normalized
adjacency with self-loops removed. With $\alpha=0$ this recovers the direct
attribution.

Both attributions, the alignment of per-window errors to per-timestep ground
truth, the figures (per-variable heatmaps, single-window case studies) and a
light qualitative *hit@k* summary live in
[`CAROTS/experiments/localization.py`](experiments/localization.py). The
module is CPU-only and consumes the `.npy` artifacts saved during inference, so
localization can be iterated on locally without a GPU.

### 4.3 How the artifacts are produced

During inference, setting `TEST.SAVE_PER_VARIABLE=True` makes the `Predictor`
save `per_variable_cd_error.npy` (windows × N) and `causality_matrix.npy` (the
learned graph) into the run's result folder
([`models/carots/predictor.py`](models/carots/predictor.py)). This flag
defaults to `False`, so upstream behavior is unchanged unless explicitly enabled.

---

## 5. Minimal, auditable changes to the upstream code

To keep the study trustworthy, the edits to upstream files are small and
inference-only:

| File | Change | Why |
|------|--------|-----|
| [`config.py`](config.py) | add `DATA.VAR_DIR` (default `'VAR'`) | point the loader at a variant folder without renaming datasets |
| [`config.py`](config.py) | add `TEST.SAVE_PER_VARIABLE` (default `False`) | opt-in to saving Step 2 artifacts |
| [`datasets/build.py`](datasets/build.py) | `VARSegLoader` reads `DATA.VAR_DIR` instead of a hard-coded `'VAR'` | load variant datasets |
| [`models/carots/scorer_carots.py`](models/carots/scorer_carots.py) | add `get_per_variable_cd_scores` | per-variable error for Step 2 |
| [`models/carots/predictor.py`](models/carots/predictor.py) | optionally save per-variable error + graph | persist Step 2 artifacts |

Everything else - the augmentors, the SOC loss, the encoder, the training loop -
is **untouched**.

The new analysis package:

```
CAROTS/experiments/
  __init__.py          # overview of the additions
  datagen.py           # generate baseline + 3 flawed VAR variants
  scenarios.py         # single source of truth for the experiment grid
  run_experiments.py   # generate (and optionally run) the run scripts
  aggregate.py         # parse results -> CSV + comparison plots + localization
  localization.py      # Step 2: per-variable attribution + figures (offline)
  run_pipeline.sh      # one-shot driver: datagen -> run all -> aggregate
  colab_carots.ipynb   # end-to-end GPU notebook (Colab alternative)
```

---

## 6. How to reproduce

> The quickest path is the one-command pipeline documented in
> [`README.md`](README.md#quick-start-gpu-cluster--workstation):
> `bash experiments/run_pipeline.sh`. The stages below explain what that script
> does. A CUDA GPU is required for training (the model hard-codes `.cuda()`);
> data generation and analysis are CPU-only.

### 6.1 Environment setup

```bash
cd CAROTS
python -m venv .venv && source .venv/bin/activate
# install torch matching your CUDA (see https://pytorch.org), then:
pip install -r requirements.txt
```

### 6.2 Data + run scripts (CPU)

```bash
# 1) generate all four data variants into data/VAR_<variant>/
python -m experiments.datagen --variant all

# 2) generate the run scripts (does not need a GPU)
python -m experiments.run_experiments
```

### 6.3 Training + evaluation (GPU)

Run everything at once:

```bash
bash experiments/generated/run_all.sh     # or: python -m experiments.run_experiments --execute
```

> `torch_geometric` and `reformer_pytorch` are imported at module-load time by
> `models/carots/encoder.py` and `layers/SelfAttention_Family.py`, so they are
> required even though the default encoder is an LSTM.
>
> To avoid retraining the causal discoverer once per anomaly type, each scenario
> shares a single causal discoverer via `CAUSAL_DISCOVERER_DIR`
> (`results/<scenario>/seed<seed>/shared_causal`); the four anomaly types of a
> scenario use the same `train.npy`, so this is exact, not an approximation.

Each run is a standard `main.py` invocation, e.g.:

```bash
python main.py \
  DATA.NAME VAR_point_global_factor2.0 \
  DATA.BASE_DIR data/ DATA.VAR_DIR VAR_nocausal \
  RESULT_DIR results/nocausal/seed2 \
  TRAIN.CHECKPOINT_DIR results/nocausal/seed2 \
  TEST.SAVE_PER_VARIABLE True \
  VISIBLE_DEVICES 0 \
  SOLVER.BASE_LR 0.001 SOLVER.WEIGHT_DECAY 0.0 SEED 2
```

> For VAR, `utils/parser.py` forces `WIN_SIZE=4`, `CUTS_PLUS.INPUT_STEP=3` and
> `SCORER.TYPE="cos"`; the localization alignment uses the same numbers.
>
> **Google Colab alternative.** If no cluster is available,
> [`experiments/colab_carots.ipynb`](experiments/colab_carots.ipynb) runs the
> same pipeline on a Colab GPU runtime (mount Drive → copy repo → install deps →
> generate → run → aggregate → copy `results/` back).

### 6.4 Aggregation + figures (CPU)

Once training has finished (`results/` is populated):

```bash
python -m experiments.aggregate
```

This writes `results/summary.csv`, `results/comparison_{auroc,auprc,f1}.png`, and
the Step 2 figures under `results/localization/<scenario>/seed<seed>/`.

---

## 7. Results

> The numbers below are filled in after the Colab run. The harness produces them
> automatically into `results/summary.csv` and the comparison plots.

### 7.1 Step 1 - detection robustness

Mean AUROC / AUPRC / F1 per (scenario × anomaly type), averaged over seeds:

| Scenario | PG | PC | CT | CG |
|----------|----|----|----|----|
| baseline | _ | _ | _ | _ |
| nocausal | _ | _ | _ | _ |
| nonstationary | _ | _ | _ | _ |
| contaminated | _ | _ | _ | _ |

Comparison chart: `results/comparison_auroc.png`.

### 7.2 Step 2 - localization (qualitative)

For each run we produce, per anomaly type:

- `*_direct_heatmap.png` / `*_causal_heatmap.png` - per-variable score over test
  windows next to the ground-truth anomalous-variable mask.
- `*_direct_window<k>.png` / `*_causal_window<k>.png` - a single-window case
  study; truly anomalous variables are highlighted in red.

A light *hit@k* sanity summary (fraction of anomalous windows whose top-k scored
variables include a truly anomalous one) is printed during aggregation. This is
a qualitative signal, not a benchmark metric.

---

## 8. Hypotheses and expected findings

- **`nocausal`** - the causal-discrepancy term loses its discriminative power
  (no parents to forecast from), so detection should rely mostly on the
  contrastive term and degrade, especially for contextual/collective anomalies.
- **`nonstationary`** - the single learned graph is wrong for most of the
  timeline; forecasting error becomes noisy everywhere, inflating false
  positives and lowering AUPRC.
- **`contaminated`** - training-set anomalies pull the "normal" centroid and the
  forecaster toward anomalous patterns, shrinking the gap between normal and
  anomalous scores at test time.
- **Localization** - the `direct` attribution should already rank truly
  anomalous variables highly for point anomalies; the `causal` propagation
  should help when anomalies sit on upstream variables whose children also
  light up, by concentrating evidence on the source.

These are pre-registered expectations; the Colab run confirms or refutes them.

---

## 9. Limitations and future work

- **Synthetic only.** VAR data isolates causal effects cleanly but is simpler
  than real telemetry. A natural extension is to repeat the study on a real
  benchmark with partial causal annotations.
- **Localization is qualitative.** We deliberately keep Step 2 visual. A
  quantitative localization benchmark (e.g. variable-level AUROC against
  `*_varlabels.npy`, or root-cause hit@k across graph depths) is a clear next
  step and the artifacts already support it.
- **Single union graph for non-stationary data.** Detecting *which* regime is
  active and switching graphs online is an interesting follow-up that would
  directly address assumption A2.
- **Propagation depth.** The causal attribution currently uses one-hop child
  evidence; multi-hop diffusion along the graph is a straightforward extension.

---

## 10. References

- Kim et al., *Causality-Aware Contrastive Learning for Robust Multivariate
  Time-Series Anomaly Detection*, ICML 2025. arXiv:2506.03964.
- Official code: https://github.com/kimanki/CAROTS
- CUTS+ causal discoverer (used internally by CAROTS).
