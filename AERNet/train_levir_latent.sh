#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=slurm_aernet_latent_seed_${1}-%j.out








# ===================================================================
# Job Execution
# ===================================================================
SEED=${1:-42}

echo "Job started on $(hostname) at $(date)"
echo "Running AERNet Benchmark (LEVIR) with SEED: ${SEED}"

nvidia-smi
# load module
module purge
module load python/3.11
module load gcc arrow/21.0.0
module load gcc opencv/4.12.0
module load scipy-stack/2025a

echo "Modules loaded."
# activate virtual environment
source AER/bin/activate

echo "Virtual environment activated."

# Train

echo "Starting Python training script..."

echo "----------------------------------------"
echo " Starting training with levir latent now..."
echo "----------------------------------------"
python train_aernet_levir_latent.py --seed ${SEED}

echo "Job finished at $(date)"