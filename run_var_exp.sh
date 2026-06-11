#!/usr/bin/env bash
set -euo pipefail

TOTAL_EPOCHS="${TOTAL_EPOCHS:-50}"
START_EPOCH="${START_EPOCH:-1}"
MAX_WORKERS_EPOCH="${MAX_WORKERS_EPOCH:-40}"
MAX_WORKERS_SYNAPSE="${MAX_WORKERS_SYNAPSE:-30}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FIXED_CLUS_SPIKE_GEN_SEEDS="${FIXED_CLUS_SPIKE_GEN_SEEDS:-60 61 62}"

if [[ ! -f "L5b_simulation.py" ]]; then
  echo "ERROR: run this script from the NeuronWithNetworkx repo root." >&2
  exit 1
fi

run_var() {
  local var_suffix="$1"
  local fixed_clus_spike_gen_seeds=()
  read -r -a fixed_clus_spike_gen_seeds <<< "${FIXED_CLUS_SPIKE_GEN_SEEDS}"

  echo "Running ${var_suffix}: epochs ${START_EPOCH}..$((START_EPOCH + TOTAL_EPOCHS - 1))"

  common_args=(
    L5b_simulation.py
    --simu_cond invivo
    --sec_type basal apical
    --spat_cond clus distr
    --dis_to_root 1
    --channel_suffix "${var_suffix}"
    --aff_mode custom
    --aff_list 0 12 24 36 48 60 72
    --num_syn_per_clus 72
    --num_epochs "${TOTAL_EPOCHS}"
    --start_epoch "${START_EPOCH}"
    --max_workers_epoch "${MAX_WORKERS_EPOCH}"
    --max_workers_synapse "${MAX_WORKERS_SYNAPSE}"
  )

  if [[ "$var_suffix" == "bgtimevar" ]]; then
    # Vary bg spike timing across epochs (bg_spike_gen_seed defaults to epoch).
    # Also repeat across fixed cluster-spike seeds to test fixed-seed sensitivity.
    "$PYTHON_BIN" "${common_args[@]}" \
      --bg_syn_pos_seed 42 \
      --clus_syn_pos_seed 42 \
      --clus_spike_gen_seed "${fixed_clus_spike_gen_seeds[@]}"
  elif [[ "$var_suffix" == "spktimevar" ]]; then
    # Vary cluster stimulus times across epochs (clus_spike_gen_seed defaults to epoch).
    "$PYTHON_BIN" "${common_args[@]}" \
      --bg_syn_pos_seed 42 \
      --clus_syn_pos_seed 42 \
      --bg_spike_gen_seed 6
  elif [[ "$var_suffix" == "bgposvar" ]]; then
    # Vary background synapse layout/weights across epochs (bg_syn_pos_seed defaults to epoch).
    "$PYTHON_BIN" "${common_args[@]}" \
      --clus_syn_pos_seed 42 \
      --bg_spike_gen_seed 6 \
      --clus_spike_gen_seed 60
  elif [[ "$var_suffix" == "clusposvar" ]]; then
    # Vary cluster layout / perm across epochs (clus_syn_pos_seed defaults to epoch).
    "$PYTHON_BIN" "${common_args[@]}" \
      --bg_syn_pos_seed 42 \
      --bg_spike_gen_seed 6 \
      --clus_spike_gen_seed 60
  else
    echo "ERROR: unknown var suffix: ${var_suffix}" >&2
    exit 1
  fi
}

run_var bgtimevar
run_var spktimevar
run_var bgposvar
run_var clusposvar
