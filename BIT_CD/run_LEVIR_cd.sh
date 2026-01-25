#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --output=slurm_latent_bit_LEVIR_seed_${1}-%j.out

# ===================================================================
# Job Execution
# ===================================================================
SEED=${1:-42}
echo "Job started on $(hostname) at $(date)"

echo "Running with SEED: ${SEED}"
nvidia-smi

# load module
module purge
module load python/3.11
module load gcc arrow/16.1.0
module load gcc opencv/4.12.0
module load scipy-stack/2024a

echo "Modules loaded."
# activate virtual environment
source bit/bin/activate

echo "Virtual environment activated."

# Train

echo "Starting Python training script: train_latent_bit.py"

python train_latent_bit_LEVIR.py --seed ${SEED}


echo "Job finished at $(date)"