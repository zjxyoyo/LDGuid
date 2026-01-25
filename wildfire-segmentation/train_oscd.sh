#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out








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
source wildfire/bin/activate

echo "Virtual environment activated."

# Train

echo "Starting Python training script..."

echo "----------------------------------------"
echo "Autoencoder training finished. Starting training..."
echo "----------------------------------------"
python -m src.train.OSCD_train_unet_with_latent

echo "Job finished at $(date)"