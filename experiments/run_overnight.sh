#!/usr/bin/env bash
# Overnight run: close the gaps that the existing cluster results still have.
#
# What is already done (as of the results reviewed on 2026-07-27)
# --------------------------------------------------------------
#   baseline       22 seeds   nonstationary   5 seeds
#   nocausal       22 seeds   contaminated    5 seeds
#   toy_chain       1 seed    toy_chain_mid / _leaf   none at all
#
# So Step 1 already has solid error bars for two scenarios, and retraining those
# seeds would burn the night for nothing. What is genuinely missing is, in order
# of value per GPU-hour:
#
# Stage 1 (minutes) : toy_chain_mid + toy_chain_leaf, anomaly on the chain
#                     middle and leaf instead of the root. Without these,
#                     "the attribution just always answers variable 0" cannot be
#                     ruled out - the single most attackable point in Step 2.
# Stage 2 (minutes) : extra seeds for all three toys, so the Step-2 metrics get
#                     error bars too. Still cheap because p=5.
# Stage 3 (hours)   : extra seeds for nonstationary + contaminated, the two
#                     underpowered Step-1 scenarios (5 seeds vs 22).
# Stage 4 (hours)   : fill the 14 missing nocausal cells (mostly
#                     collective_global for seeds 8-20). Lowest value: that cell
#                     already has 9 seeds and a std of 0.001.
#
# Every stage aggregates immediately afterwards and nothing uses `set -e`, so
# whatever has finished by morning is usable even if the machine dies midway.
#
# Usage (from the CAROTS/ directory):
#   bash experiments/run_overnight.sh
#
# Environment overrides:
#   PYTHON=python3.10  bash experiments/run_overnight.sh   # pick the interpreter
#   GPU=1              bash experiments/run_overnight.sh   # use CUDA device 1
#   TOY_SEEDS="2 3 4"  bash experiments/run_overnight.sh   # seeds for the toys
#   STEP1_SEEDS="4 5"  bash experiments/run_overnight.sh   # extra Step-1 seeds
#   ONLY=toys          bash experiments/run_overnight.sh   # stages 1-2 only
#   ONLY=step1         bash experiments/run_overnight.sh   # stages 3-4 only
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
BASE_SEED="${BASE_SEED:-2}"
TOY_SEEDS="${TOY_SEEDS:-3 4}"
STEP1_SEEDS="${STEP1_SEEDS:-4 5 6 7}"
ONLY="${ONLY:-all}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

TOYS="toy_chain toy_chain_mid toy_chain_leaf"
# nonstationary/contaminated are the ones short on seeds; nocausal only needs
# its missing cells backfilled, which --skip-existing handles on its own.
UNDERPOWERED="nonstationary contaminated"

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
log "   python      : $PYTHON"
log "   GPU         : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "   toy seeds   : $BASE_SEED $TOY_SEEDS"
log "   step-1 seeds: $STEP1_SEEDS"
log "   stages      : $ONLY"
log "   full log    : $LOG"
log "=============================================="

# Generate scripts and run them for one (scenario list, seed list) block.
# --skip-existing means a rerun of this script never retrains a finished cell,
# so it is safe to just start it again after a crash.
run_block() {
  local label="$1" scenarios="$2" seeds="$3"
  step "generate run scripts: $label" \
    "$PYTHON" -m experiments.run_experiments \
      --scenarios $scenarios --seeds $seeds --keep-going --skip-existing
  for scenario in $scenarios; do
    step "train+eval: $scenario ($label)" \
      bash "experiments/generated/run_${scenario}.sh"
  done
}

# --------------------------------------------------------------------------
# Stage 1 - the controls that Step 2 is missing entirely (cheap: p=5)
# --------------------------------------------------------------------------
if [ "$ONLY" != "step1" ]; then
  log ""
  log ">>> STAGE 1/4: toy controls - anomaly on chain middle and leaf"

  for variant in $TOYS; do
    ensure_data "$variant"
  done

  run_block "seed $BASE_SEED" "toy_chain_mid toy_chain_leaf" "$BASE_SEED"

  # Aggregate now: this is the part of the talk that must exist by morning.
  step "aggregate after stage 1" \
    "$PYTHON" -m experiments.aggregate --localization-scenarios $TOYS
  log ">>> STAGE 1 done - the 'always variable 0' objection is now answerable"

  # ------------------------------------------------------------------------
  # Stage 2 - error bars on the Step-2 metrics, still only minutes
  # ------------------------------------------------------------------------
  log ""
  log ">>> STAGE 2/4: extra seeds for all three toys ($TOY_SEEDS)"
  run_block "seeds $TOY_SEEDS" "$TOYS" "$TOY_SEEDS"
  step "aggregate after stage 2" \
    "$PYTHON" -m experiments.aggregate --localization-scenarios $TOYS
  log ">>> STAGE 2 done - Step 2 now has seed variability"
fi

# --------------------------------------------------------------------------
# Stage 3 - lift the two underpowered Step-1 scenarios from 5 seeds
# --------------------------------------------------------------------------
if [ "$ONLY" != "toys" ]; then
  for variant in baseline nocausal nonstationary contaminated; do
    ensure_data "$variant"
  done

  # One complete seed at a time, aggregating after each, so a night that ends
  # early still leaves a consistent table.
  for seed in $STEP1_SEEDS; do
    log ""
    log ">>> STAGE 3/4: $UNDERPOWERED, seed $seed"
    run_block "seed $seed" "$UNDERPOWERED" "$seed"
    step "aggregate after seed $seed" \
      "$PYTHON" -m experiments.aggregate --no-localization
    log ">>> seed $seed complete"
  done

  # ------------------------------------------------------------------------
  # Stage 4 - backfill the missing nocausal cells. Last on purpose: lowest
  # value, and --skip-existing means only the gaps are actually trained.
  # ------------------------------------------------------------------------
  log ""
  log ">>> STAGE 4/4: backfill missing nocausal cells"
  run_block "backfill" "nocausal" "$(seq -s' ' 0 20) 42"
  step "aggregate after stage 4" "$PYTHON" -m experiments.aggregate --no-localization
  log ">>> STAGE 4 done"
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
