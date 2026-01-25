#!/bin/bash 
#SBATCH --account=def-bereyhia
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --exclude=fc10512

module purge
module load python/3.11
module load gcc arrow/17.0.0
module load scipy-stack/2024a
module load gcc opencv/4.12.0

echo "Modules loaded."

source wildfire/bin/activate
echo "Virtual environment activated."

echo "----------------------------------------"
echo " Starting training with latent..."
echo "----------------------------------------"
python -m src.train.LDGtrain