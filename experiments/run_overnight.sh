#!/usr/bin/env bash
# Overnight run: finish the Step 2 localization story, then add seeds to Step 1.
#
# Why a separate script from run_pipeline.sh
# ------------------------------------------
# run_pipeline.sh runs the Step-1 grid once, on one seed. That is enough for a
# first look but leaves two gaps that a reviewer will immediately probe:
#
#   * Step 1 has no error bars, so a scenario difference cannot be told apart
#     from seed-to-seed noise.
#   * Step 2 was only ever run with the anomaly on the chain *root*, so
#     "the attribution always answers variable 0" is not ruled out and the
#     causal-propagation term is never actually put to the test.
#
# This script closes both gaps, ordered by value per GPU-hour, and it keeps
# going if a single run fails. Every stage aggregates immediately afterwards, so
# whatever has finished by morning is already usable - even if the machine dies
# halfway through.
#
# Stage 1 (minutes)  : the three toy variants (p=5, so the causal discoverer is
#                      cheap). Anomalies on the chain root / middle / leaf.
# Stage 2 (hours)    : one extra seed for the whole Step-1 grid -> mean +/- std.
# Stage 3 (hours)    : a second extra seed, if the night is long enough.
#
# Usage (from the CAROTS/ directory):
#   bash experiments/run_overnight.sh
#
# Environment overrides:
#   PYTHON=python3.10  bash experiments/run_overnight.sh   # pick the interpreter
#   GPU=1              bash experiments/run_overnight.sh   # use CUDA device 1
#   SEEDS="3 4"        bash experiments/run_overnight.sh   # extra Step-1 seeds
#   SKIP_TOYS=1        bash experiments/run_overnight.sh   # only do the seeds
#   SKIP_SEEDS=1       bash experiments/run_overnight.sh   # only do the toys
#
# Deliberately no `set -e`: one crashed run must not throw away the night.
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-3 4}"
BASE_SEED="${BASE_SEED:-2}"
SKIP_TOYS="${SKIP_TOYS:-0}"
SKIP_SEEDS="${SKIP_SEEDS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

mkdir -p results
LOG="results/overnight_log.txt"
FAILURES=0

log() { echo "$*" | tee -a "$LOG"; }

# Regenerate a data variant only if it is missing. Overwriting existing data
# would silently change the process under an already-finished seed and make the
# seeds incomparable - exactly the noise the extra seeds are meant to measure.
ensure_data() {
  local variant="$1"
  if [ -f "data/VAR_${variant}/train.npy" ]; then
    log "    data/VAR_${variant} exists - keeping it (not regenerating)"
    return 0
  fi
  step "datagen: $variant (missing)" "$PYTHON" -m experiments.datagen --variant "$variant"
}

# Run a command, time it, record the outcome, and carry on regardless.
step() {
  local label="$1"; shift
  local started elapsed status
  started=$(date +%s)
  log ""
  log "--- [$(date '+%F %T')] $label"
  "$@" >>"$LOG" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - started ))
  if [ $status -eq 0 ]; then
    log "    OK   ($((elapsed / 60))m $((elapsed % 60))s)"
  else
    FAILURES=$((FAILURES + 1))
    log "    FAIL (exit $status after $((elapsed / 60))m) - continuing"
  fi
  return 0
}

log "=============================================="
log " CAROTS overnight run - started $(date '+%F %T')"
log "   python     : $PYTHON"
log "   GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "   extra seeds: $SEEDS"
log "   full log   : $LOG"
log "=============================================="

# --------------------------------------------------------------------------
# Stage 1 - Step 2 controls on the toy chain (cheap: p=5)
# --------------------------------------------------------------------------
if [ "$SKIP_TOYS" != "1" ]; then
  log ""
  log ">>> STAGE 1/3: toy-chain localization controls"

  for variant in toy_chain toy_chain_mid toy_chain_leaf; do
    ensure_data "$variant"
  done

  step "generate toy run scripts" \
    "$PYTHON" -m experiments.run_experiments \
      --scenarios toy_chain toy_chain_mid toy_chain_leaf --seeds "$BASE_SEED"

  for scenario in toy_chain toy_chain_mid toy_chain_leaf; do
    step "train+eval: $scenario (seed $BASE_SEED)" \
      bash "experiments/generated/run_${scenario}.sh"
  done

  # Aggregate now: the Step 2 numbers are the part of the talk that must exist.
  step "aggregate after stage 1" "$PYTHON" -m experiments.aggregate
  log ">>> STAGE 1 done - results/localization_metrics.csv is populated"
fi

# --------------------------------------------------------------------------
# Stages 2+ - extra seeds for the Step-1 grid, one complete seed at a time
# --------------------------------------------------------------------------
if [ "$SKIP_SEEDS" != "1" ]; then
  stage=2
  for seed in $SEEDS; do
    log ""
    log ">>> STAGE $stage: Step-1 grid, seed $seed"

    # Only fills in what a fresh Colab session is missing; existing data is left
    # exactly as the earlier seeds saw it.
    for variant in baseline nocausal nonstationary contaminated; do
      ensure_data "$variant"
    done

    step "generate Step-1 run scripts (seed $seed)" \
      "$PYTHON" -m experiments.run_experiments --seeds "$seed"

    for scenario in baseline nocausal nonstationary contaminated; do
      step "train+eval: $scenario (seed $seed)" \
        bash "experiments/generated/run_${scenario}.sh"
    done

    # Aggregate after every completed seed so a partial night still yields a
    # consistent table with error bars over however many seeds finished.
    step "aggregate after seed $seed" "$PYTHON" -m experiments.aggregate
    log ">>> STAGE $stage done - seed $seed complete"
    stage=$((stage + 1))
  done
fi

log ""
log "=============================================="
log " Finished $(date '+%F %T') with $FAILURES failed step(s)"
log " Outputs:"
log "   results/summary.csv                     per-run metrics"
log "   results/summary_by_seed.csv             mean/std over seeds"
log "   results/comparison_auroc.png            now with error bars"
log "   results/localization_metrics.csv        Step 2, quantitative"
log "   results/localization_alpha_sweep.csv    does propagation help?"
log "   results/localization/<scenario>/seed*/  figures"
log "=============================================="
exit 0
