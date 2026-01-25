#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out








# ===================================================================
# Job Execution
# ===================================================================
echo "Job started on $(hostname) at $(date)"

nvidia-smi
SEED=$1

# load module
module purge
module load python/3.11
module load gcc arrow/21.0.0
module load gcc opencv/4.12.0
module load scipy-stack/2025a

echo "Modules loaded."
# activate virtual environment
source wildfire/bin/activate

echo "Virtual environment activated."

# Train

echo "Starting job for SEED = ${SEED}"

echo "----------------------------------------"
echo " Starting training..."
echo "----------------------------------------"
python -m src.train.svcd_train_unet_with_latent ${SEED}

echo "Job finished at $(date)"