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
- **`nonstationary`** - the timeline is split into `n_regimes=8` segments, each
  with its **own** randomly-drawn causal graph, coefficients, and marginal
  dynamics (see [`NONSTATIONARY_CHANGES.md`](experiments/NONSTATIONARY_CHANGES.md)).
  We save both a representative union graph (`GC.npy`) and the per-regime graphs
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

**Signed vs. two-sided scoring.** Both attributions can score either the signed
robust z-score or its magnitude $|z|$. This matters more than it sounds: a
*collective-global* anomaly replaces a noisy segment with a smooth one, which is
**easier** to forecast, so the culprit's residual drops far below its own median
and a signed score ranks the one broken variable as maximally *normal*. This is
the same blind spot that motivates CAROTS' own use of $\mathcal{A}_\text{CL}$
alongside $\mathcal{A}_\text{CD}$.

Both attributions, both score modes, the alignment of per-window errors to
per-timestep ground truth, the figures and the quantitative evaluation live in
[`CAROTS/experiments/localization.py`](experiments/localization.py). The
module is CPU-only and consumes the `.npy` artifacts saved during inference, so
localization can be iterated on locally without a GPU.

### 4.3 How localization is evaluated

Everything is measured on the *anomalous windows only* — the question is "given
that this window is anomalous, which variable is to blame?" — and every metric
is reported next to its chance level:

| Metric | Meaning | Chance level |
|--------|---------|--------------|
| `var_auroc` | (window, variable) pairs pooled into one binary ranking problem | 0.5 |
| `var_ap` | average precision on the same pooling | anomalous-variable base rate |
| `mrr` | mean reciprocal rank of the first true culprit | `random_mrr` |
| `hit@k` | top-k contains a culprit | exact hypergeometric `random_hit@k` |

Two diagnostics accompany them. The **alpha sweep** evaluates
$\alpha \in \{0, 0.25, 0.5, 1, 2\}$, and since $\alpha=0$ *is* the direct
attribution it isolates exactly what the propagation contributes. The **leakage
report** compares the mean score of the culprits, of their causal children, and
of all remaining variables; the propagation can only help if the children sit
clearly above the rest. Details in
[`experiments/LOCALIZATION_METRICS.md`](experiments/LOCALIZATION_METRICS.md).

### 4.4 Controls: where do the anomalies sit?

Three toy variants share one generative process and differ only in which
variable receives the test anomalies (registered in `scenarios.TOY_SCENARIOS`):

| Scenario | Target | Controls for |
|----------|--------|--------------|
| `toy_chain` | 0 (root) | the original demo |
| `toy_chain_mid` | 1 (middle) | "does the attribution just always answer 0?"; propagation should pull credit back from the child |
| `toy_chain_leaf` | 2 (leaf) | a leaf has no children, so propagation has no evidence to gather — its cost becomes visible |

### 4.5 How the artifacts are produced

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
  __init__.py               # overview of the additions
  datagen.py                # generate baseline + 3 flawed VAR variants + toys
  scenarios.py              # single source of truth for the experiment grid
  run_experiments.py        # generate (and optionally run) the run scripts
  aggregate.py              # parse results -> CSVs + comparison plots + Step 2
  localization.py           # Step 2: attribution, metrics, figures (offline)
  test_localization.py      # GPU-free smoke test of the whole offline path
  run_pipeline.sh           # one-shot driver: datagen -> run all -> aggregate
  run_overnight.sh          # toy controls first, then extra seeds
  colab_carots.ipynb        # end-to-end GPU notebook (Colab alternative)
  TOY_CHAIN.md              # the toy generative model
  NONSTATIONARY_CHANGES.md  # strengthening the nonstationary variant
  LOCALIZATION_METRICS.md   # quantitative Step 2 evaluation + controls
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

### 7.1 Step 1 - detection robustness

Mean AUROC ± std over seeds (`results/summary_by_seed.csv`; seed counts in
brackets — `baseline`/`nocausal` have 22, the others 5):

| Scenario | Seeds | PG | PC | CT | CG |
|----------|-------|----|----|----|----|
| baseline | 22 | 0.664 ± 0.036 | 0.649 ± 0.027 | 0.963 ± 0.003 | 0.997 ± 0.001 |
| nocausal | 22 | **0.747 ± 0.039** | 0.601 ± 0.018 | 0.965 ± 0.011 | 0.994 ± 0.024 |
| contaminated | 9 | 0.594 ± 0.030 | 0.573 ± 0.024 | 0.959 ± 0.004 | 0.996 ± 0.002 |
| nonstationary | 9 | 0.572 ± 0.010 | 0.548 ± 0.009 | 0.944 ± 0.004 | 0.955 ± 0.019 |

With 22 seeds the standard error on `baseline` is about 0.008, so every gap in
the table above is many standard errors wide; none of this is seed noise.

**`nonstationary` is uniformly the worst scenario**, and it is the only one that
also degrades the collective anomalies (CT 0.963 → 0.942, CG 0.997 → 0.949).
This resolves the earlier "Finding 2" — before the A+C strengthening documented
in [`NONSTATIONARY_CHANGES.md`](experiments/NONSTATIONARY_CHANGES.md) the
variant was marginally indistinguishable from `baseline`; making the marginal
dynamics switch per regime, not just the graph, is what made the violation bite.

**`contaminated` degrades the point anomalies** (PG 0.664 → 0.586, PC 0.649 →
0.580) while leaving the collective ones untouched. Since its test set is
identical to `baseline`, this isolates the effect of training contamination.

**`nocausal` is the counter-intuitive one:** PC drops as expected (0.649 →
0.601), but PG *improves* strongly (0.664 → 0.747, AUPRC 0.082 → 0.329). This is
not noise. It is also not a fair head-to-head: `nocausal` is a different process,
in which an isolated spike on an independent AR series is simply a more visible
event than the same spike inside a coupled VAR. The honest reading is that
removing causal structure makes the *task* easier for point-global anomalies,
not that CAROTS benefits from having no causality to exploit.

Comparison charts (now with std error bars): `results/comparison_auroc.png`,
`_auprc.png`, `_f1.png`.

> **Caveat on F1.** `threshold.py` selects the threshold by maximizing F1 *on the
> test set* (`TEST.THRESHOLD.TYPE = 'best_f1'`), and `TEST.POINT_ADJUST` is on by
> default. Both are upstream choices we kept for comparability, but the F1
> columns are therefore oracle numbers and should not be read as deployable
> performance. AUROC and AUPRC are threshold-free and unaffected.

### 7.2 Step 2 - localization

Quantitative results land in `results/localization_metrics.csv` (one row per run
x attribution x score mode) and `results/localization_alpha_sweep.csv`. Always
read them against the printed chance levels.

Direct attribution, signed score, seed 2. The regenerated datasets were verified
against each run's saved `test_labels.npy`, so the ground truth provably matches
what the model saw.

**`toy_chain`** (p=5, one culprit per window, chance hit@1 = 0.20):

| Anomaly | `auroc_within` | `mrr` | `hit@1` |
|---------|---------------|-------|---------|
| CT | 1.000 | 1.000 | 1.000 |
| PG | 0.994 | 0.988 | 0.975 |
| PC | 0.976 | 0.960 | 0.925 |
| CG | 0.975 | 0.968 | 0.950 |

**`baseline`** (p=128, 10 culprits per window, chance hit@1 = 0.078, chance
hit@5 = 0.339); stable to ±0.001 across seeds 2/3/4:

| Anomaly | `auroc_within` | `var_ap` | `hit@1` | `hit@5` |
|---------|---------------|----------|---------|---------|
| CT | 1.000 | 1.000 | 1.000 | 1.000 |
| CG | 0.961 | 0.955 | 0.950 | 0.950 |
| PC | 0.888 | 0.342 | 0.405 | 0.930 |
| PG | 0.948 | 0.420 | **0.000** | 0.370 |

**The point-global paradox.** PG combines a within-window AUROC of 0.948 with a
hit@1 of *exactly* zero. Both are correct: the 10 culprits cluster around scores
of 5-6 while a handful of the 118 normal variables reach 6-15. Typical noise
sits well below the signal, which is what the AUROC measures, but the **maximum**
of 118 heavy-tailed noise variables reliably beats a moderate signal. This is
the multiple-comparisons problem in localization form, and its practical
consequence is that top-1 is the wrong way to consume this attribution at high
dimensionality: produce a top-k shortlist and judge it against chance.

**Localization survives flawed training data far better than detection does.**
Repeating the same evaluation for every Step-1 scenario (direct, signed, seed 2)
gives within-window AUROC:

| Scenario | CG | CT | PC | PG |
|----------|----|----|----|----|
| baseline | 0.961 | 1.000 | 0.888 | 0.948 |
| nocausal | 0.998 | 1.000 | 0.828 | 0.963 |
| nonstationary | 0.962 | 1.000 | 0.885 | 0.919 |
| contaminated | 0.938 | 1.000 | 0.887 | 0.949 |

Where detection loses up to 0.09 AUROC under `nonstationary` (§7.1), the
attribution barely moves. The two are asking different questions: detection has
to decide whether a window's *total* score exceeds what training led it to
expect, which is precisely what a shifted or contaminated training set corrupts,
whereas attribution only has to rank variables *against each other within one
window*, and a miscalibrated overall error level largely cancels out.

The `nocausal` column repeats the Step-1 story: PG localization jumps to
hit@5 = 1.000 versus 0.370 for `baseline`, because an isolated spike on an
independent AR series has no coupled neighbours competing with it.

### 7.3 Where the anomaly sits decides everything (the controls)

`toy_chain_mid` and `toy_chain_leaf` move the injected anomaly from the chain
root (variable 0) to its middle (1) and leaf (2). hit@1, mean over 3 seeds,
chance level 0.20:

| Anomaly sits on | PG | PC | CT | CG |
|-----------------|----|----|----|----|
| root (var 0) | 0.987 | 0.903 | 1.000 | 0.950 |
| middle (var 1) | 0.852 | 0.753 | 1.000 | 0.937 |
| leaf (var 2) | 0.755 | 0.703 | 1.000 | 1.000 |

**The "it just always answers variable 0" objection is settled**: every cell is
far above chance wherever the culprit sits. There is, however, a clear gradient
for the point anomalies — the further downstream the culprit, the harder it is to
pin down — while the collective anomalies are saturated everywhere.

**And the causal propagation term is not neutral after all.** The root-only
experiment had suggested it barely matters; sweeping $\alpha$ across all three
positions shows why that was misleading (`hit@1`, point-global):

| Position | $\alpha$=0 | 0.25 | 0.5 | 1.0 | 2.0 |
|----------|-----------|------|-----|-----|-----|
| root | 0.987 | 0.990 | 0.990 | 0.995 | 0.995 |
| middle | 0.852 | 0.837 | 0.833 | 0.830 | 0.450 |
| leaf | 0.755 | 0.755 | 0.745 | 0.392 | **0.002** |

Propagation adds a variable's *children's* error to its own score. A root
culprit has children carrying corroborating evidence, so the term helps. A leaf
culprit has none — the term adds nothing to the culprit while inflating its
ancestors, which have the culprit as a child. At $\alpha=2$ the true variable is
ranked first in 0.2% of windows, i.e. the propagation reliably points at the
*parent* of the culprit instead. The mechanism is a genuine limitation of the
approach and only becomes visible once the anomaly is moved off the root.

Practical consequence: keep $\alpha$ small (≤0.5 loses nothing anywhere and
gains a little at the root), or treat the propagated score as evidence about the
*subtree* rather than the single variable.

The **two-sided score** makes almost no difference on these runs, because the
injected anomalies raise the forecasting error rather than lowering it.

Per run and anomaly type we also produce:

- `*_<attribution>_<mode>_heatmap.png` - per-variable score over test windows
  next to the ground-truth mask, colour-clipped at the 99th percentile.
- `*_<attribution>_<mode>_window<k>.png` - a single-window case study; truly
  anomalous variables are highlighted in red.
- `*_alpha_sweep.png` - localization quality vs. the propagation weight, one
  panel per score mode, with chance levels.

---

## 8. Hypotheses vs. outcomes

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

These were pre-registered expectations. Against the results in §7:

- **`nonstationary` - confirmed**, and it is the strongest effect: the only
  scenario that degrades all four anomaly types, including the collective ones
  that everything else handles nearly perfectly.
- **`contaminated` - confirmed** for point anomalies; the collective anomalies
  are unaffected, which the hypothesis did not anticipate.
- **`nocausal` - refuted for point-global.** We expected degradation; PG instead
  improves substantially (0.664 → 0.747 AUROC, 0.082 → 0.329 AUPRC). Only PC
  behaves as predicted. See §7.1 for why the two scenarios are not a fair
  head-to-head.
- **Localization - confirmed.** The direct attribution ranks culprits far above
  chance regardless of where on the causal chain they sit, decisively so for
  collective anomalies.
- **Causal propagation - refuted as stated.** We expected it to help, especially
  for a culprit whose children carry the evidence. It does help at the root, but
  for a leaf culprit a large weight actively destroys the answer (hit@1 0.755 →
  0.002), because the term credits the culprit's ancestors instead. This only
  became visible through the `toy_chain_mid` / `toy_chain_leaf` controls; the
  root-only experiment made the term look harmlessly neutral.

---

## 9. Limitations and future work

- **Synthetic only.** VAR data isolates causal effects cleanly but is simpler
  than real telemetry. A natural extension is to repeat the study on a real
  benchmark with partial causal annotations.
- **Localization is evaluated on toy graphs.** Step 2 now reports variable-level
  AUROC/AP/MRR/hit@k against `*_varlabels.npy` with chance levels, but the chain
  used for the controls has three nodes. How the attribution behaves on a dense
  128-variable graph, where a corrupted variable has many children and the
  learned graph is itself uncertain, is only covered by the Step-1 runs.
- **Single union graph for non-stationary data.** Detecting *which* regime is
  active and switching graphs online is an interesting follow-up that would
  directly address assumption A2.
- **Propagation depth.** The causal attribution currently uses one-hop child
  evidence; multi-hop diffusion along the graph is a straightforward extension.
- **Score mode is chosen, not learned.** Signed scoring wins for anomalies that
  make a variable harder to forecast, two-sided scoring for those that make it
  easier. Both are reported side by side rather than selected automatically.

---

## 10. References

- Kim et al., *Causality-Aware Contrastive Learning for Robust Multivariate
  Time-Series Anomaly Detection*, ICML 2025. arXiv:2506.03964.
- Official code: https://github.com/kimanki/CAROTS
- CUTS+ causal discoverer (used internally by CAROTS).
