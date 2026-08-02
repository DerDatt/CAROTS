# Step 2 – Quantitative localization + controls (changes)

Date: 2026-07-26
Files touched: `experiments/localization.py`, `experiments/aggregate.py`,
`experiments/scenarios.py`, `experiments/datagen.py`,
`experiments/run_overnight.sh` (new), `experiments/test_localization.py` (new)

This document records *every* change made in this round and why. It is the
companion to [`TOY_CHAIN.md`](TOY_CHAIN.md) (what the toy data is) and
[`NONSTATIONARY_CHANGES.md`](NONSTATIONARY_CHANGES.md) (the previous round).

**Nothing about training, the model or the losses changed.** As before, Step 2
is pure inference-time post-processing on the `.npy` artifacts a run writes when
`TEST.SAVE_PER_VARIABLE=True`.

---

## 1. Problem this round addresses

The first localization pass had four weaknesses:

| # | Weakness | Consequence |
|---|----------|-------------|
| 1 | Only a "light qualitative" hit@k, no chance level | Could not say whether the attribution beat guessing. |
| 2 | The causal propagation (`alpha`) was never measured | Its contribution was unproven. |
| 3 | Test anomalies always sat on chain variable `0` | "The attribution just always answers 0" could not be ruled out. |
| 4 | Heatmap colour scale spanned the raw error range | One extreme window saturated the scale; the plots were solid black. |

A fifth issue showed up in the existing `collective_global` results: the
attribution ranked the true culprit **last**. Section 4 explains why and what
was done about it.

---

## 2. Quantitative metrics (`localization.py`)

`localization_report` was rewritten from a hit@k sanity print into a proper
evaluation. All metrics are computed **on the truly anomalous windows only** —
Step 2 answers "given that this window is anomalous, which variable is to
blame?", so normal windows would only dilute the numbers.

| Metric | Meaning | Chance level reported alongside |
|--------|---------|---------------------------------|
| `auroc_within` | Mean of the **per-window** AUROC: are the culprits separated from the other variables *of the same window*? | `0.5` |
| `var_auroc` | All (window, variable) pairs of the anomalous windows pooled into one binary ranking problem. | `0.5` |
| `var_ap` | Average precision on the same pooling. | `random_ap` = anomalous-variable base rate |
| `mrr` | Mean reciprocal rank of the *first* truly anomalous variable. `1.0` = culprit always ranked first. | `random_mrr` |
| `hit@k` | Fraction of anomalous windows whose top-k contains a culprit. | `random_hit@k`, computed exactly via the hypergeometric miss probability (`_random_hit_at_k`) |

Every metric ships with its own chance level, because with `p=5` toy variables a
hit@3 of 0.6 is *exactly* what guessing gives.

`auroc_within` exists because pooling can mislead: a "loud" window where every
variable scores high would outrank a quiet one, inflating `var_auroc` without
the culprits being separated from their own window's normal variables. It is
computed from ranks in Mann-Whitney form, so no per-window `sklearn` call is
needed. On the runs measured so far the two agree closely, which is itself worth
knowing — see §11 for the case where they agree and *both* still mislead.

`roc_auc_score` / `average_precision_score` come from scikit-learn, which the
repo already depends on (`threshold.py`). The import is guarded, so the module
still loads without it (those two metrics then return `NaN`).

## 3. Alpha sweep and leakage diagnostic (`localization.py`)

**`alpha_sweep(...)`** evaluates the attribution across propagation weights
(default `0, 0.25, 0.5, 1.0, 2.0`). Since `alpha=0` *is* the direct attribution,
this isolates the propagation term. A flat curve means the propagation adds
nothing on that dataset — which is a result to report, not to hide.

**`leakage_report(...)`** splits the variables of each anomalous window into
three groups and returns their mean score:

* `culprit` – the truly anomalous variables,
* `children` – their causal children (which inherit a corrupted input),
* `other` – everything else.

This is the precondition for the propagation to have anything to work with: it
can only help if `children` sits clearly above `other`. If the children look
like unrelated variables, there is no leakage to undo, and a flat alpha sweep is
then *explained* rather than merely observed.

**`plot_alpha_sweep(...)`** renders the sweep as one panel per score mode, with
dotted chance lines.

## 4. Two-sided score (`localize_direct`, `localize_causal`)

### The failure we found

In the existing `toy_chain` results, `collective_global` ranked the true culprit
**last of five** with an all-negative score profile. The reason is structural,
not a bug:

* the attribution is a robust z-score of the forecasting error,
* a collective-global anomaly replaces a noisy segment with a smooth/shifted
  one, which is **easier** to forecast one step ahead,
* so the culprit's residual falls far *below* its own median, giving a large
  **negative** z-score,
* a signed score therefore reads "maximally normal" for the one variable that is
  actually broken.

This is worth stating in its own right: it is the same reason CAROTS needs the
contrastive score $\mathcal{A}_\text{CL}$ next to the forecasting score
$\mathcal{A}_\text{CD}$ — a forecasting residual alone is blind to anomalies
that make a signal *more* predictable.

### The fix

Both attribution functions gained a `two_sided` flag. When set, the score is
`|robust z|`, i.e. the *magnitude* of the deviation from the variable's typical
error rather than its signed value. `run()` now evaluates the full cross product

    {direct, causal} x {signed, two_sided}

so the two independent design choices (propagate along the graph; score
magnitude instead of sign) can be judged separately. The keys in the returned
summary are `direct_signed`, `direct_two_sided`, `causal_signed`,
`causal_two_sided`.

The smoke test pins down the expected behaviour, including its limit: a drop is
bounded below by zero error, so `|z|` stays small for the culprit while a
spiking causal child is unbounded and can still outrank it. Two-sided scoring
fixes the *sign* problem; closing the remaining gap is what propagation is for.

## 4b. Representative case-study window (`pick_example_window`)

The old implementation preferred a window with the fewest culprits. When every
anomalous window has exactly one culprit — which is the case for the toys —
`argmin` silently returned the **first** anomalous window. That is usually the
least typical one: the start of an injected segment, before the disturbance has
taken effect.

This produced a badly misleading figure. The `collective_global` case study for
`toy_chain` showed the true culprit ranked *last of five*, suggesting a total
failure, while over all 200 anomalous windows that same attribution actually
achieves hit@1 = 0.95.

`pick_example_window` now optionally takes the score matrix and returns the
window whose culprit rank is the **median** — the honest middle of the
distribution rather than a lucky or unlucky extreme. `run()` picks it once from
the plain signed attribution so all four case-study figures stay comparable.

## 5. Readable heatmaps (`plot_error_heatmap`)

The colour range is now clipped to the 99th percentile of the scores
(`clip_percentile`, configurable) and the clipping is stated in the plot title.
Forecasting errors span several orders of magnitude, so without this a single
extreme window drives `vmax` and everything else renders as black.

## 6. Toy control scenarios (`scenarios.py`, `datagen.py`)

`TOY_SCENARIOS` in `scenarios.py` is now the single source of truth for which
variable each toy variant corrupts:

| Scenario | Injection target | What it controls for |
|----------|------------------|----------------------|
| `toy_chain` | `0` (chain root) | The original demo. |
| `toy_chain_mid` | `1` (chain middle) | Rules out "the attribution always answers 0"; the case where propagation should pull credit back from the child. |
| `toy_chain_leaf` | `2` (chain leaf) | A leaf has no causal children, so propagation has no evidence to gather. If the propagated score is *worse* here, that is the price of the propagation term. |

The generative process is identical across the three (see `TOY_CHAIN.md`); only
the injection target differs. Because `p=5`, the causal discoverer is cheap and
all three fit in minutes of GPU time.

`datagen.py` changes:

* `--variant` accepts `toy_chain_mid`, `toy_chain_leaf` and `toys`
  (= generate all toy variants).
* `--toy-root-var` now defaults to `None`, meaning "use the target registered
  for this variant in `TOY_SCENARIOS`". Passing it explicitly still overrides.
* The two `simulate_toy_chain` progress prints were made ASCII-only. They used
  `→`, which crashes `datagen` on a Windows console (cp1252 cannot encode it).

`run_experiments.py`: the "which anomaly factor grid applies" check changed from
`scenario == "toy_chain"` to `scenario in TOY_SCENARIOS`, so all toys correctly
use factor 3.0.

## 7. Aggregation outputs (`aggregate.py`)

| New output | Content |
|------------|---------|
| `results/summary_by_seed.csv` | Mean / std / count per (scenario, anomaly type). With one seed the std is `NaN`, which is the honest answer. |
| `results/localization_metrics.csv` | One row per (run, attribution, score mode) with all Step 2 metrics plus the leakage diagnostic. |
| `results/localization_alpha_sweep.csv` | One row per (run, score mode, alpha). |

`comparison_{auroc,auprc,f1}.png` now draw std error bars as soon as more than
one seed is present, and the title states how many seeds went in.

## 8. Overnight driver (`run_overnight.sh`, new)

`run_pipeline.sh` is unchanged. The new script exists because a single night has
to be spent in the right order:

1. **Stage 1 (minutes):** the three toy variants — this completes the Step 2
   story and is cheap.
2. **Stage 2 (hours):** one extra seed across the whole Step-1 grid, which is
   what turns the scenario comparison into something with error bars.
3. **Stage 3:** a second extra seed if there is time.

It aggregates after every stage, so a partial night still yields a consistent
table, and it deliberately omits `set -e` so one crashed run at 3 AM does not
discard everything after it. Timings and failures go to
`results/overnight_log.txt`.

## 9. Smoke test (`test_localization.py`, new)

A GPU-free test that fabricates the artifacts a real run would write and checks
that the whole offline path works before hours of cluster time are spent on it:

1. `localization.run` end to end, figures included;
2. a planted error *spike* on the root is recovered (`hit@1 = 1.0`, well above
   `random_hit@1 = 0.2`);
3. a planted error *drop* breaks the signed score (`var_auroc = 0.0`) and is
   repaired by the two-sided score (`var_auroc = 0.85`);
4. `aggregate` walks a miniature results tree and builds all three CSVs.

Run it with:

```bash
python -m experiments.test_localization
```

---

## 10. First results on the existing cluster runs

Computed on the artifacts already produced by the cluster (no retraining), with
`direct` attribution and the signed score, seed 2. The regenerated datasets were
verified against each run's saved `test_labels.npy` — the derived window labels
match bit for bit, so `datagen` is deterministic and the ground truth is valid.

**`toy_chain` (p=5, one culprit per window, chance hit@1 = 0.20):**

| Anomaly | `auroc_within` | `mrr` | `hit@1` |
|---------|---------------|-------|---------|
| collective_trend | 1.000 | 1.000 | 1.000 |
| point_global | 0.994 | 0.988 | 0.975 |
| point_contextual | 0.976 | 0.960 | 0.925 |
| collective_global | 0.975 | 0.968 | 0.950 |

**`baseline` (p=128, 10 culprits per window, chance hit@1 = 0.078,
chance hit@5 = 0.339), stable to ±0.001 across seeds 2/3/4:**

| Anomaly | `auroc_within` | `var_ap` | `hit@1` | `hit@5` |
|---------|---------------|----------|---------|---------|
| collective_trend | 1.000 | 1.000 | 1.000 | 1.000 |
| collective_global | 0.961 | 0.955 | 0.950 | 0.950 |
| point_contextual | 0.888 | 0.342 | 0.405 | 0.930 |
| point_global | 0.948 | 0.420 | **0.000** | 0.370 |

**Across all Step-1 scenarios** (within-window AUROC, direct, signed, seed 2):

| Scenario | CG | CT | PC | PG |
|----------|----|----|----|----|
| baseline | 0.961 | 1.000 | 0.888 | 0.948 |
| nocausal | 0.998 | 1.000 | 0.828 | 0.963 |
| nonstationary | 0.962 | 1.000 | 0.885 | 0.919 |
| contaminated | 0.938 | 1.000 | 0.887 | 0.949 |

Detection loses up to 0.09 AUROC under `nonstationary`; attribution barely
moves. Attribution only ranks variables against each other inside one window, so
a miscalibrated overall error level largely cancels out.

### 11. The point-global paradox

`point_global` at `p=128` has a within-window AUROC of 0.948 — a random culprit
beats a random normal variable 95% of the time — and yet `hit@1` is *exactly*
0.000 across all 200 anomalous windows and all three seeds, with `hit@5` at 0.37
against a chance level of 0.339.

Both numbers are correct and consistent. Inspecting individual windows shows the
10 culprits clustered at scores around 5–6 while a handful of the 118 normal
variables reach 6–15. Typical noise sits well below the signal (hence the high
AUROC), but the **maximum** of 118 heavy-tailed noise variables reliably exceeds
a moderate signal. Roughly five normal variables sit above the culprit cluster
in most windows, which reproduces both the AUROC and the hit@k numbers.

This is the multiple-comparisons problem in localization form, and it has a
practical consequence: **top-1 is the wrong way to consume this attribution at
high dimensionality.** Report a top-k shortlist, and read `hit@k` against its
chance level rather than trusting an impressive-looking AUROC.

## 12. How to use this

```bash
# CPU only, on artifacts you already have - no retraining needed:
python -m experiments.aggregate

# GPU, overnight:
bash experiments/run_overnight.sh
```

The first command alone already produces the quantitative Step 2 table and the
alpha sweep from existing runs. The overnight script is only needed for the
toy controls and the extra seeds.
