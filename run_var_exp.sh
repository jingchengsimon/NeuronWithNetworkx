#!/usr/bin/env bash
set -euo pipefail

TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-30}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "L5b_simulation.py" ]]; then
  echo "ERROR: run this script from the NeuronWithNetworkx repo root." >&2
  exit 1
fi

run_batches() {
  local var_suffix="$1"

  for spat_cond in clus distr; do
    local start=1

    while [[ "$start" -le "$TOTAL_EPOCHS" ]]; do
      local remain=$((TOTAL_EPOCHS - start + 1))
      local epochs_this_batch="$BATCH_SIZE"
      if [[ "$remain" -lt "$BATCH_SIZE" ]]; then
        epochs_this_batch="$remain"
      fi

      echo "Running ${var_suffix}/${spat_cond}: epochs ${start}..$((start + epochs_this_batch - 1))"

      common_args=(
        L5b_simulation.py
        --simu_cond invivo
        --sec_type basal apical
        --spat_cond "${spat_cond}"
        --dis_to_root 1
        --channel_suffix "${var_suffix}"
        --aff_mode custom
        --aff_list 0 12 24 36 48 60 72
        --num_syn_per_clus 72
        --epoch_mode multi
        --num_batches 1
        --epochs_per_batch "${epochs_this_batch}"
        --start_epoch "${start}"
      )

      if [[ "$var_suffix" == "bgtimevar" ]]; then
        "$PYTHON_BIN" "${common_args[@]}" \
          --bg_syn_pos_seed 42 \
          --clus_syn_pos_seed 42 \
          --clus_spike_gen_seed 60
      elif [[ "$var_suffix" == "spktimevar" ]]; then
        "$PYTHON_BIN" "${common_args[@]}" \
          --bg_syn_pos_seed 42 \
          --clus_syn_pos_seed 42 \
          --bg_spike_gen_seed 6
      elif [[ "$var_suffix" == "bgposvar" ]]; then
        "$PYTHON_BIN" "${common_args[@]}" \
          --clus_syn_pos_seed 42 \
          --bg_spike_gen_seed 6 \
          --clus_spike_gen_seed 60
      elif [[ "$var_suffix" == "clusposvar" ]]; then
        "$PYTHON_BIN" "${common_args[@]}" \
          --bg_syn_pos_seed 42 \
          --bg_spike_gen_seed 6 \
          --clus_spike_gen_seed 60
      else
        echo "ERROR: unknown var suffix: ${var_suffix}" >&2
        exit 1
      fi

      start=$((start + epochs_this_batch))
    done
  done
}

# run_batches bgtimevar
# run_batches spktimevar
run_batches bgposvar
run_batches clusposvar
