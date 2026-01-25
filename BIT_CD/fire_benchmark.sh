#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=slurm_latent_bit-%j.out

# ===================================================================
# Environment Setup
# ===================================================================
echo "Job started on $(hostname) at $(date)"

nvidia-smi
# load module
module purge
module load python/3.11
module load gcc arrow/21.0.0
module load scipy-stack/2025a
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
# If no argument is given, it will default to 'Fire'.
data_name=${1:-Fire}
echo "Training on dataset: ${data_name}"

net_G=base_transformer_pos_s4_dd8
batch_size=8
lr=0.01
max_epochs=200 # <-- 200 epochs
project_name=BIT_${data_name}_benchmark

# --- Run Training ---
# 
python fire_cd.py \
    --data_name ${data_name} \
    --net_G ${net_G} \
    --max_epochs ${max_epochs} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --project_name ${project_name} \
    --gpu_ids 0

echo "Job finished at $(date)"