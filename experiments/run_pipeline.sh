#!/usr/bin/env bash
# End-to-end pipeline for the CAROTS flawed-data + localization study.
#
# Runs the whole thing on a GPU machine, in order:
#   1. generate the 4 VAR data variants   (CPU, fast)
#   2. generate the run scripts           (CPU, fast)
#   3. train + evaluate every scenario    (GPU, the slow part)
#   4. aggregate metrics + figures        (CPU, fast)
#
# Usage (from the CAROTS/ directory):
#   bash experiments/run_pipeline.sh
#
# Environment overrides:
#   PYTHON=python3.10   bash experiments/run_pipeline.sh   # pick the interpreter
#   GPU=1               bash experiments/run_pipeline.sh   # use CUDA device 1
#
# For a cluster scheduler (e.g. SLURM), wrap this script in your job file and
# request one GPU, e.g.:  srun --gres=gpu:1 bash experiments/run_pipeline.sh
set -euo pipefail

# Run from the repository root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

echo "=============================================="
echo " CAROTS study pipeline"
echo "   python : $PYTHON"
echo "   GPU    : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=============================================="

echo
echo ">>> [1/4] Generating the four VAR data variants ..."
"$PYTHON" -m experiments.datagen --variant all

echo
echo ">>> [2/4] Generating the run scripts ..."
"$PYTHON" -m experiments.run_experiments

echo
echo ">>> [3/4] Training + evaluating all scenarios (this is the long step) ..."
bash experiments/generated/run_all.sh

echo
echo ">>> [4/4] Aggregating results + figures ..."
"$PYTHON" -m experiments.aggregate

echo
echo "=============================================="
echo " Done. See:"
echo "   results/summary.csv"
echo "   results/comparison_auroc.png / _auprc.png / _f1.png"
echo "   results/localization/<scenario>/seed<seed>/"
echo "=============================================="
