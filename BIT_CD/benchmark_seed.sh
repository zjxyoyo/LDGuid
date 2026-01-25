#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=slurm_latent_bit-%j.out

# ===================================================================
# Job Execution
# ===================================================================
echo "Job started on $(hostname) at $(date)"

nvidia-smi
# load module
module purge
module load python/3.11
module load gcc arrow/16.1.0
module load scipy-stack/2024a

echo "Modules loaded."
# activate virtual environment
source bit/bin/activate

echo "Virtual environment activated."

# ===================================================================
# Job Execution
# ===================================================================
echo "Starting Python training script for the original BIT model..."

# --- Configuration ---
# Use the first command-line argument as the dataset name.
# If no argument is given, it will default to 'LEVIR'.
data_name=${1:-LEVIR}
echo "Training on dataset: ${data_name}"

net_G=base_transformer_pos_s4_dd8
batch_size=8
lr=0.01
max_epochs=200

# --- Define Seeds ---
SEEDS="7 42 123 1000 2024"

# --- Loop over seeds ---
for seed in $SEEDS
do
    echo "-----------------------------------------"
    echo "Starting run with SEED: ${seed}"
    echo "-----------------------------------------"

    # --- Modify project_name to include the seed ---

    project_name=CD_${net_G}_${data_name}_seed${seed}

    # --- Run Training ---
    python main_cd_v2.py \
        --data_name ${data_name} \
        --net_G ${net_G} \
        --max_epochs ${max_epochs} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --project_name ${project_name} \
        --gpu_ids 0 \
        --seed ${seed} 

    echo "--- Finished run with SEED: ${seed} ---"
done

echo "All seed runs finished at $(date)"