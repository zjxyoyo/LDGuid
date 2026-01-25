#!/bin/bash
#SBATCH --account=def-bereyhia
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out








# ===================================================================
# Job Execution
# ===================================================================
echo "Job started on $(hostname) at $(date)"

nvidia-smi
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

echo "Starting Python training script..."

echo "----------------------------------------"
echo " Starting joint training..."
echo "----------------------------------------"
python -m src.ablation.joint_train

echo "Job finished at $(date)"