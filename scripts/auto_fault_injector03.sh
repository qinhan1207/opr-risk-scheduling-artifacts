#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# Sanitized reconstructed TC/NetEm injection script for OPR-v3 dataset
# -----------------------------------------------------
# Purpose:
#   Apply repeatable controlled network perturbation patterns for
#   short-term placement-risk prediction experiments on the testbed.
#
# Notes:
#   1. This script is reconstructed and sanitized from earlier test scripts.
#   2. It is intended to reproduce comparable risk patterns, not to claim
#      byte-level identity with the original machine-specific script.
#   3. Hostnames, paths, and private environment settings should be replaced
#      before release.
# =====================================================

# ================= Config =================
CONTAINER_NAME="${CONTAINER_NAME:-member3-control-plane}"
INTERFACE="${INTERFACE:-eth0}"

# Total running time in hours. Each scenario lasts roughly 60-80 seconds.
TOTAL_HOURS="${TOTAL_HOURS:-4}"

# Enable slight randomization for duration and netem parameters.
ENABLE_RANDOMIZE="${ENABLE_RANDOMIZE:-1}"

# Fixed seed for reproducibility. Leave empty to use current time.
SEED="${SEED:-}"

# Output directory.
OUT_DIR_BASE="${OUT_DIR_BASE:-runs}"

# Risk threshold used by the dataset label construction.
TAU_NET="${TAU_NET:-45}"

# ANSI colors.
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ================= Utilities =================
now_ms() {
  date +%s%3N
}

rand_init() {
  if [[ -z "${SEED}" ]]; then
    SEED=$(date +%s)
  fi
  RANDOM=$((SEED % 32768))
}

# Integer in [min, max].
rand_int() {
  local min=$1
  local max=$2

  if [[ "$max" -le "$min" ]]; then
    echo "$min"
    return
  fi

  echo $((min + (RANDOM % (max - min + 1))))
}

choose_profile() {
  # Weighted random selection.
  # The weights roughly reflect the released v3 trace composition:
  # stable, gradual, microspike, volatility, abrupt, recovery.
  local r
  r=$(rand_int 1 100)

  if [[ "$r" -le 25 ]]; then
    echo "stable"
  elif [[ "$r" -le 46 ]]; then
    echo "gradual"
  elif [[ "$r" -le 66 ]]; then
    echo "microspike"
  elif [[ "$r" -le 86 ]]; then
    echo "volatility"
  elif [[ "$r" -le 96 ]]; then
    echo "abrupt"
  else
    echo "recovery"
  fi
}

ensure_tools() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is not available."
    exit 1
  }

  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" || {
    echo "Container does not exist or is not running: $CONTAINER_NAME"
    exit 1
  }
}

mkdir_run_dir() {
  mkdir -p "$OUT_DIR_BASE"

  RUN_ID=$(date +%Y%m%d_%H%M%S)
  OUT_DIR="$OUT_DIR_BASE/$RUN_ID"
  mkdir -p "$OUT_DIR"

  LOG_FILE="$OUT_DIR/injection_log.csv"
  META_FILE="$OUT_DIR/run_meta.txt"

  echo "timestamp_ms,rel_sec,run_id,scenario_id,block_type,phase,is_risk_phase,op,delay_ms,jitter_ms,loss_pct,tau_net,container,iface" > "$LOG_FILE"

  {
    echo "run_id=$RUN_ID"
    echo "container=$CONTAINER_NAME"
    echo "iface=$INTERFACE"
    echo "total_hours=$TOTAL_HOURS"
    echo "enable_randomize=$ENABLE_RANDOMIZE"
    echo "seed=$SEED"
    echo "tau_net=$TAU_NET"
    echo "script_note=sanitized reconstructed TC/NetEm profiles for OPR-v3 dataset"
  } > "$META_FILE"
}

START_MS=0

log_event() {
  local scenario_id="$1"
  local block_type="$2"
  local phase="$3"
  local is_risk_phase="$4"
  local op="$5"
  local delay="$6"
  local jitter="$7"
  local loss="$8"

  local t
  t=$(now_ms)

  local rel_ms=$((t - START_MS))
  local rel_sec
  rel_sec=$(awk -v ms="$rel_ms" 'BEGIN{printf "%.3f", ms / 1000.0}')

  echo "${t},${rel_sec},${RUN_ID},${scenario_id},${block_type},${phase},${is_risk_phase},${op},${delay},${jitter},${loss},${TAU_NET},${CONTAINER_NAME},${INTERFACE}" >> "$LOG_FILE"
}

cleanup() {
  echo -e "\n${YELLOW}[cleanup] removing TC rules...${NC}"
  docker exec "$CONTAINER_NAME" tc qdisc del dev "$INTERFACE" root >/dev/null 2>&1 || true
  echo -e "${GREEN}[cleanup] done.${NC}"
}
trap cleanup EXIT

apply_rule() {
  # apply_rule <scenario_id> <block_type> <phase> <is_risk_phase 0/1> <delay_ms> <jitter_ms> <loss_pct>
  local scenario_id="$1"
  local block_type="$2"
  local phase="$3"
  local is_risk_phase="$4"
  local delay="$5"
  local jitter="$6"
  local loss="$7"

  local op="replace"

  log_event "$scenario_id" "$block_type" "$phase" "$is_risk_phase" "$op" "$delay" "$jitter" "$loss"

  local cmd="docker exec $CONTAINER_NAME tc qdisc replace dev $INTERFACE root netem"

  if [[ "$delay" -gt 0 ]]; then
    cmd="$cmd delay ${delay}ms"
    if [[ "$jitter" -gt 0 ]]; then
      cmd="$cmd ${jitter}ms distribution normal"
    fi
  fi

  if [[ "$loss" -gt 0 ]]; then
    cmd="$cmd loss ${loss}%"
  fi

  eval "$cmd"
}

wait_progress() {
  local seconds="$1"
  local start
  start=$(date +%s)

  local end=$((start + seconds))

  while [[ "$(date +%s)" -lt "$end" ]]; do
    local left=$((end - $(date +%s)))
    printf "\r  remaining %3d s" "$left"
    sleep 1
  done

  echo ""
}

run_phase() {
  # run_phase <scenario_id> <block_type> <phase> <is_risk_phase> <delay> <jitter> <loss> <duration>
  local scenario_id="$1"
  local block_type="$2"
  local phase="$3"
  local is_risk_phase="$4"
  local delay="$5"
  local jitter="$6"
  local loss="$7"
  local duration="$8"

  echo -e "${YELLOW}  phase=${phase}, delay=${delay}ms, jitter=${jitter}ms, loss=${loss}%, duration=${duration}s${NC}"

  apply_rule "$scenario_id" "$block_type" "$phase" "$is_risk_phase" "$delay" "$jitter" "$loss"
  wait_progress "$duration"
}

# ================= Parameter helpers =================
stable_delay() {
  rand_int 18 22
}

stable_jitter() {
  rand_int 1 3
}

post_delay() {
  rand_int 18 22
}

post_jitter() {
  rand_int 1 3
}

risk_delay_80() {
  rand_int 75 88
}

risk_delay_90() {
  rand_int 84 96
}

risk_jitter() {
  rand_int 10 20
}

# ================= Scenario profiles =================

stable_normal_block() {
  local scenario_id="$1"
  local block_type="stable_normal_block"

  local dur
  dur=$(rand_int 45 99)

  run_phase "$scenario_id" "$block_type" "stable" 0 "$(stable_delay)" "$(stable_jitter)" 0 "$dur"
}

gradual_onset_block() {
  local scenario_id="$1"
  local block_type="gradual_onset_block"

  local d_stable d_prec d_risk d_post
  d_stable=$(rand_int 12 27)
  d_prec=$(rand_int 14 27)
  d_risk=$(rand_int 6 13)
  d_post=$(rand_int 18 44)

  run_phase "$scenario_id" "$block_type" "stable_before_gradual_onset" 0 "$(stable_delay)" "$(stable_jitter)" 0 "$d_stable"

  # A ramp-like precursor below tau_net.
  local step1=$((d_prec / 3))
  local step2=$((d_prec / 3))
  local step3=$((d_prec - step1 - step2))

  run_phase "$scenario_id" "$block_type" "precursor_gradual_onset" 0 "$(rand_int 25 29)" "$(rand_int 3 5)" 0 "$step1"
  run_phase "$scenario_id" "$block_type" "precursor_gradual_onset" 0 "$(rand_int 31 35)" "$(rand_int 4 6)" 0 "$step2"
  run_phase "$scenario_id" "$block_type" "precursor_gradual_onset" 0 "$(rand_int 39 43)" "$(rand_int 2 4)" 0 "$step3"

  run_phase "$scenario_id" "$block_type" "risk_gradual_onset" 1 "$(risk_delay_80)" "$(rand_int 10 16)" 0 "$d_risk"
  run_phase "$scenario_id" "$block_type" "post_gradual_onset" 0 "$(post_delay)" "$(post_jitter)" 0 "$d_post"
}

microspike_onset_block() {
  local scenario_id="$1"
  local block_type="microspike_onset_block"

  local d_stable d_prec d_risk d_post
  d_stable=$(rand_int 14 29)
  d_prec=$(rand_int 14 29)
  d_risk=$(rand_int 6 13)
  d_post=$(rand_int 18 44)

  run_phase "$scenario_id" "$block_type" "stable_before_microspike_onset" 0 "$(stable_delay)" "$(stable_jitter)" 0 "$d_stable"

  # Precursor with intermittent near-threshold spikes.
  local elapsed=0
  while [[ "$elapsed" -lt "$d_prec" ]]; do
    local normal_dur
    normal_dur=$(rand_int 1 3)

    if [[ $((elapsed + normal_dur)) -gt "$d_prec" ]]; then
      normal_dur=$((d_prec - elapsed))
    fi

    if [[ "$normal_dur" -gt 0 ]]; then
      run_phase "$scenario_id" "$block_type" "precursor_microspike_onset" 0 "$(rand_int 20 24)" "$(rand_int 2 4)" 0 "$normal_dur"
      elapsed=$((elapsed + normal_dur))
    fi

    if [[ "$elapsed" -ge "$d_prec" ]]; then
      break
    fi

    local spike_dur=1
    run_phase "$scenario_id" "$block_type" "precursor_microspike_onset" 0 "$(rand_int 38 44)" "$(rand_int 2 4)" 0 "$spike_dur"
    elapsed=$((elapsed + spike_dur))
  done

  run_phase "$scenario_id" "$block_type" "risk_microspike_onset" 1 "$(rand_int 78 90)" "$(rand_int 12 20)" 0 "$d_risk"
  run_phase "$scenario_id" "$block_type" "post_microspike_onset" 0 "$(post_delay)" "$(post_jitter)" 0 "$d_post"
}

volatility_onset_block() {
  local scenario_id="$1"
  local block_type="volatility_onset_block"

  local d_stable d_prec d_risk d_post
  d_stable=$(rand_int 12 27)
  d_prec=$(rand_int 14 29)
  d_risk=$(rand_int 6 13)
  d_post=$(rand_int 18 44)

  run_phase "$scenario_id" "$block_type" "stable_before_volatility_onset" 0 "$(rand_int 20 22)" "$(rand_int 1 3)" 0 "$d_stable"

  # High-variance precursor mostly below tau_net.
  run_phase "$scenario_id" "$block_type" "precursor_volatility_onset" 0 "$(rand_int 20 24)" "$(rand_int 8 14)" 0 "$d_prec"

  run_phase "$scenario_id" "$block_type" "risk_volatility_onset" 1 "$(risk_delay_80)" "$(risk_jitter)" 0 "$d_risk"
  run_phase "$scenario_id" "$block_type" "post_volatility_onset" 0 "$(rand_int 20 22)" "$(rand_int 2 4)" 0 "$d_post"
}

abrupt_onset_block() {
  local scenario_id="$1"
  local block_type="abrupt_onset_block"

  local d_stable d_risk d_post
  d_stable=$(rand_int 24 47)
  d_risk=$(rand_int 6 13)
  d_post=$(rand_int 18 44)

  run_phase "$scenario_id" "$block_type" "stable_before_abrupt_onset" 0 "$(stable_delay)" "$(stable_jitter)" 0 "$d_stable"

  # Sudden jump without an explicit precursor phase.
  run_phase "$scenario_id" "$block_type" "risk_abrupt_onset" 1 "$(risk_delay_90)" "$(risk_jitter)" 0 "$d_risk"
  run_phase "$scenario_id" "$block_type" "post_abrupt_onset" 0 "$(post_delay)" "$(post_jitter)" 0 "$d_post"
}

recovery_block() {
  local scenario_id="$1"
  local block_type="recovery_block"

  local d_risk d_decay d_post
  d_risk=$(rand_int 8 21)
  d_decay=$(rand_int 8 17)
  d_post=$(rand_int 28 59)

  run_phase "$scenario_id" "$block_type" "risk_before_recovery" 1 "$(risk_delay_90)" "$(risk_jitter)" 0 "$d_risk"

  # Decay from high risk to normal state.
  local step1=$((d_decay / 4))
  local step2=$((d_decay / 4))
  local step3=$((d_decay / 4))
  local step4=$((d_decay - step1 - step2 - step3))

  run_phase "$scenario_id" "$block_type" "decay_recovery" 1 "$(rand_int 65 75)" "$(rand_int 10 18)" 0 "$step1"
  run_phase "$scenario_id" "$block_type" "decay_recovery" 1 "$(rand_int 50 60)" "$(rand_int 8 14)" 0 "$step2"
  run_phase "$scenario_id" "$block_type" "decay_recovery" 0 "$(rand_int 38 44)" "$(rand_int 5 9)" 0 "$step3"
  run_phase "$scenario_id" "$block_type" "decay_recovery" 0 "$(rand_int 26 32)" "$(rand_int 3 6)" 0 "$step4"

  run_phase "$scenario_id" "$block_type" "post_recovery" 0 "$(post_delay)" "$(post_jitter)" 0 "$d_post"
}

run_scenario() {
  local scenario_id="$1"
  local profile="$2"

  # Reset qdisc before each scenario to avoid residual state.
  docker exec "$CONTAINER_NAME" tc qdisc del dev "$INTERFACE" root >/dev/null 2>&1 || true

  echo -e "\n${GREEN}▶ Scenario ${scenario_id}: ${profile}${NC}"

  case "$profile" in
    stable)
      stable_normal_block "$scenario_id"
      ;;
    gradual)
      gradual_onset_block "$scenario_id"
      ;;
    microspike)
      microspike_onset_block "$scenario_id"
      ;;
    volatility)
      volatility_onset_block "$scenario_id"
      ;;
    abrupt)
      abrupt_onset_block "$scenario_id"
      ;;
    recovery)
      recovery_block "$scenario_id"
      ;;
    *)
      echo "Unknown profile: $profile"
      exit 1
      ;;
  esac

  echo -e "${GREEN}✔ Scenario ${scenario_id} done.${NC}"
}

main() {
  ensure_tools
  rand_init
  mkdir_run_dir

  START_MS=$(now_ms)
  echo "start_ms=$START_MS" >> "$META_FILE"

  echo -e "${GREEN}Starting reconstructed TC/NetEm injection script.${NC}"
  echo -e "${CYAN}Output directory: ${OUT_DIR}${NC}"
  echo -e "${CYAN}Injection log: ${LOG_FILE}${NC}"
  echo -e "${CYAN}Total hours: ${TOTAL_HOURS}${NC}"
  echo -e "${CYAN}Seed: ${SEED}${NC}"
  echo -e "${CYAN}Risk threshold tau_net: ${TAU_NET} ms${NC}"

  local end_time
  end_time=$(( $(date +%s) + TOTAL_HOURS * 3600 ))

  local scenario_id=0

  while [[ "$(date +%s)" -lt "$end_time" ]]; do
    local profile
    profile=$(choose_profile)

    run_scenario "$scenario_id" "$profile"
    scenario_id=$((scenario_id + 1))
  done

  echo -e "\n${GREEN}All scenarios completed.${NC}"
  echo -e "${CYAN}Next step: collect RTT traces and construct sliding-window meta samples with tau_net=${TAU_NET}.${NC}"
}

main "$@"
