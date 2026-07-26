# CAROTS study: flawed training data + variable-level localization

This repository is a fork of the official **CAROTS** implementation
([kimanki/CAROTS](https://github.com/kimanki/CAROTS), ICML 2025) with two
research extensions added on top:

- **Step 1 — Robustness study.** Stress-test CAROTS on three kinds of *flawed
  training data* (no causal relationship, non-stationary causal relationship,
  and anomalies in the training set) against a clean baseline, using synthetic
  VAR datasets where the causal ground truth is known.
- **Step 2 — Variable-level anomaly localization.** Extend CAROTS so it reports
  *which* variable is anomalous, not just *when* — **without changing training,
  the model, or the loss functions**.

All added code lives in [`experiments/`](experiments/) plus a few small,
clearly-marked hooks in the upstream files. The full methodology, design
rationale, and the list of upstream changes are in
[`REPORT.md`](REPORT.md) — **read that first** to understand what is going on.

---

## Quick start (GPU cluster / workstation)

> **A CUDA GPU is required.** The model hard-codes `.cuda()`, so training will
> not run on CPU. Data generation and result aggregation are CPU-only.
> Tested with Python 3.9+.

### 1. Set up the environment

```bash
cd CAROTS

# create an isolated environment (venv shown; conda works too)
python -m venv .venv
source .venv/bin/activate

# install PyTorch matching YOUR cluster's CUDA first — see https://pytorch.org
# (example for CUDA 12.1:)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# then the rest of the dependencies
pip install -r requirements.txt
```

### 2. Run everything with one command

From the `CAROTS/` directory:

```bash
bash experiments/run_pipeline.sh
```

This generates the data, trains + evaluates all scenarios on the GPU, and writes
the metrics and figures. To pick a specific GPU or interpreter:

```bash
GPU=1 PYTHON=python3.10 bash experiments/run_pipeline.sh
```

On a SLURM cluster, wrap it in your job script and request one GPU:

```bash
srun --gres=gpu:1 bash experiments/run_pipeline.sh
```

### 3. Where the results land

| Output | Path |
|--------|------|
| Metrics table (AUROC / AUPRC / F1 per scenario × anomaly) | `results/summary.csv` |
| Same, averaged over seeds with std | `results/summary_by_seed.csv` |
| Comparison bar charts (error bars once >1 seed) | `results/comparison_auroc.png`, `_auprc.png`, `_f1.png` |
| Step 2 localization metrics | `results/localization_metrics.csv` |
| Step 2 causal-propagation sweep | `results/localization_alpha_sweep.csv` |
| Step 2 localization figures | `results/localization/<scenario>/seed<seed>/` |

### 4. Longer run: controls and extra seeds

`run_pipeline.sh` covers one seed of the Step-1 grid. To also get the Step 2
controls and error bars, use the overnight driver — it does the cheap,
high-value work first and aggregates after every stage, so an interrupted run
still leaves a consistent set of results:

```bash
bash experiments/run_overnight.sh          # toys, then seeds 3 and 4
SKIP_SEEDS=1 bash experiments/run_overnight.sh   # only the toy controls (minutes)
```

Progress, per-step timings and failures are logged to
`results/overnight_log.txt`. Before spending GPU time, you can verify the whole
offline analysis path locally:

```bash
python -m experiments.test_localization
```

---

## Running it step by step (optional)

If you want to run or inspect the stages individually (all from `CAROTS/`):

```bash
# 1) generate the 4 VAR variants into data/VAR_<variant>/  (CPU, fast)
python -m experiments.datagen --variant all

# 2) generate the run scripts into experiments/generated/  (CPU, fast)
python -m experiments.run_experiments

# 3a) train + evaluate EVERYTHING (GPU, slow)
bash experiments/generated/run_all.sh
#    ... or run a single scenario first as a sanity check:
bash experiments/generated/run_baseline.sh
#    ... or let the Python harness execute the runs directly:
python -m experiments.run_experiments --execute

# 4) aggregate metrics + figures  (CPU, fast)
python -m experiments.aggregate
```

A single training invocation is a standard `main.py` call, e.g.:

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

Override `VISIBLE_DEVICES` (or set `CUDA_VISIBLE_DEVICES`) to choose the GPU.

---

## What's in `experiments/`

```
experiments/
  datagen.py               # generate baseline + 3 flawed VAR variants + toys
  scenarios.py             # single source of truth for the experiment grid
  run_experiments.py       # generate (and optionally --execute) the run scripts
  run_pipeline.sh          # one-shot: datagen -> run all -> aggregate
  run_overnight.sh         # toy controls first, then extra seeds, resumable
  aggregate.py             # parse results -> CSVs + comparison plots + Step 2
  localization.py          # Step 2: attribution, metrics, figures (offline, CPU)
  test_localization.py     # GPU-free smoke test of the offline analysis
  colab_carots.ipynb       # alternative: run on Google Colab instead of a cluster

  TOY_CHAIN.md             # the toy causal-chain data
  NONSTATIONARY_CHANGES.md # strengthening the nonstationary variant
  LOCALIZATION_METRICS.md  # quantitative Step 2 evaluation + controls
```

---

## Tips & troubleshooting

- **Quick GPU smoke test** (verify the whole pipeline in ~a few minutes before
  committing to the full grid) — generate the baseline data, then run one
  scenario with tiny epoch counts:

  ```bash
  python -m experiments.datagen --variant baseline
  python main.py \
    DATA.NAME VAR_point_global_factor2.0 \
    DATA.BASE_DIR data/ DATA.VAR_DIR VAR_baseline \
    RESULT_DIR results_smoke/baseline/seed2 \
    TRAIN.CHECKPOINT_DIR results_smoke/baseline/seed2 \
    TEST.SAVE_PER_VARIABLE True \
    SOLVER.MAX_EPOCH 2 WARMUP_EPOCHS 0 CUTS_PLUS.SOLVER.MAX_EPOCH 2 \
    SEED 2
  ```

- **`ModuleNotFoundError: torch_geometric` / `reformer_pytorch`** — these are
  imported at import time even with the default LSTM encoder. Make sure
  `pip install -r requirements.txt` finished successfully.
- **Speed** — the causal discoverer (CUTS+) is the bottleneck. Each scenario
  reuses one shared causal discoverer across its four anomaly types (via
  `CAUSAL_DISCOVERER_DIR`), which the generated scripts already set for you.
- **`DATA_LOADER.NUM_WORKERS`** — defaults to 4; lower it if your node has few
  CPU cores (`... DATA_LOADER.NUM_WORKERS 2`).

For full methodology and the exact list of upstream code changes, see
[`REPORT.md`](REPORT.md).

---

<details>
<summary><b>Upstream CAROTS documentation (original README)</b></summary>

# [ICML 2025] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection

## Overview
The figures below illustrate the overview pipeline and components of the CAROTS framework. 
For more detailed information, please refer to our paper.

![Overview](./figure/overview.png)  

![Component](./figure/components.png)  


---

## Dataset Preparation

### Download Links
- [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
- [WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
- [PSM](https://github.com/eBay/RANSynCoders/tree/main)
- [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master)
- [SMAP](https://github.com/khundman/telemanom)


For the `SMD` and `SMAP` datasets, preprocessing was performed based on the script available at [OmniAnomaly's data_preprocess.py](https://github.com/NetManAIOps/OmniAnomaly/blob/master/data_preprocess.py). After preprocessing, the datasets were placed in the following directories:

```
data/
├── ServerMachineDataset/  # Preprocessed SMD dataset
└── SMAP_MSL/              # Preprocessed SMAP dataset
```

### Synthetic Dataset Generation
To generate synthetic datasets (Lorenz96 and VAR), use the following commands:

```bash
cd data/Lorenz96
python generate.py
```

```bash
cd data/VAR
python generate.py
```

### Dataset Organization
Place the downloaded datasets in the `data/` directory, ensuring each dataset resides in its respective subdirectory:

```
data/
├── Lorenz96/
├── VAR/
├── SWaT/
├── WADI/
├── PSM/
├── ServerMachineDataset/
└── SMAP_MSL/
```

---

## Example Execution

Scripts for running the model are located in the `scripts/` directory. Use the following command to execute the model:

```bash
bash scripts/{dataset}.sh
```

**Example:** To run the model on the `SWaT` dataset:

```bash
bash scripts/SWaT.sh
```

---

The results of the model execution are saved in the `results/{dataset}` directory. 

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{kim2025causality,
  title={Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection},
  author={Kim, HyunGi and Mok, Jisoo and Lee, Dongjun and Lew, Jaihyun and Kim, Sungjae and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2506.03964},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. For commercial use, permission is required.

---

## Acknowledgements

Please provide proper attribution if you use our codebase.  
If you use our work, kindly cite our paper as mentioned in the Citation section.

</details>