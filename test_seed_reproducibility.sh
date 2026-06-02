#!/usr/bin/env bash
# Quick smoke tests before run_var_exp.sh batch jobs.
# Requires NEURON + repo dependencies; run from repo root.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${SEED_TEST_ROOT:-./tmp_seed_repro_test}"

if [[ ! -f "L5b_simulation.py" ]]; then
  echo "ERROR: run this script from the NeuronWithNetworkx repo root." >&2
  exit 1
fi

echo "Seed reproducibility smoke test"
echo "  python:       ${PYTHON_BIN}"
echo "  results_root: ${RESULTS_ROOT}"
echo
echo "Checks (seed-controlled inputs in section_synapse_df):"
echo "  1) max_workers_synapse 1 vs 50 (same four seeds)"
echo "  2) max_workers_epoch 50 with 2 parallel epochs (seeds fixed, not epoch)"
echo "  3) aff_mode=custom with different worker settings"
echo "  4) each seed changes only its domain:"
echo "       bg_syn_pos_seed -> layout/weights/cluster; bg_spike_gen_seed -> bg spikes;"
echo "       clus_syn_pos_seed -> cluster layout/perm; clus_spike_gen_seed -> cluster spikes"
echo
echo "Note: seed checks use section_synapse_df fingerprints; soma_v is printed for info only"
echo "      (default atol=1 mV if you add manual soma_v checks: --soma_v_atol 1.0)"
echo

"${PYTHON_BIN}" scripts/test_seed_reproducibility.py \
  --results_root "${RESULTS_ROOT}" \
  --python "${PYTHON_BIN}"

echo
echo "Smoke test passed. You can start the batch experiment:"
echo "  bash run_var_exp.sh"
echo
echo "Optional overrides for the batch:"
echo "  MAX_WORKERS_EPOCH=50 MAX_WORKERS_SYNAPSE=30 TOTAL_EPOCHS=100 START_EPOCH=1 bash run_var_exp.sh"
