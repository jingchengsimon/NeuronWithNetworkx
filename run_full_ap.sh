#!/bin/bash
# Canonical reproduction script: submit ALL 1200 tasks as 6 SLURM jobs.
#   1200 = 100 epochs x 2 sec_type x 3 dis_to_root x 2 spat_cond(clus,distr)
# Run with no args on the login node -> fans out into 6 jobs (one per
# sec_type x dis_to_root combo), each running clus+distr x 100 epochs = 200 tasks.
#SBATCH --partition=mem-redhat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00

if [ -z "$SLURM_JOB_ID" ]; then
  cd ~/projects/l5b_functional_clustering
  for sec in basal apical; do
    for r in 0 1 2; do
      sbatch --job-name=full_${sec}_r${r} \
        --output=slurm_jobs/full_${sec}_r${r}_%j.out \
        --error=slurm_jobs/full_${sec}_r${r}_%j.err \
        "$0" "$sec" "$r"
    done
  done
  exit 0
fi

# --- inside a SLURM job: $1 = sec_type, $2 = dis_to_root ---
source activate l5b_amarel_neuron
cd ~/projects/l5b_functional_clustering
python L5b_simulation.py \
  --results_root ~/projects/l5b_functional_clustering/results/simulation_singclus_supple_Jun26_ap \
  --simu_cond invivo \
  --sec_type "$1" \
  --spat_cond clus distr \
  --dis_to_root "$2" \
  --channel_suffix singclus \
  --num_clusters 1 \
  --num_syn_per_clus 72 \
  --aff_mode custom \
  --aff_list 0 12 24 36 48 60 72 \
  --with_ap \
  --folder_tag 1 \
  --start_epoch 1 \
  --num_epochs 100 \
  --max_workers_epoch 40 \
  --max_workers_synapse 50
