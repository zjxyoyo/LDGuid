#!/bin/bash 
#SBATCH --account=def-bereyhia
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --exclude=fc10512

# --- MODIFIED: Job Array ---
#SBATCH --array=0-6

# --- MODIFIED: Unique output log for each job ---
#SBATCH --output=slurm-%A_%a.out

# --- 1. 定义你的参数数组 ---
ALPHAS=(0.00001 0.001 0.01 1.2 1.5 2 5)

# --- 2. 获取这个特定作业的索引 ---
# SLURM_ARRAY_TASK_ID 会是 0, 1, 2, ... 6
TASK_ID=$SLURM_ARRAY_TASK_ID

# --- 3. 从数组中选择这个作业对应的alpha值 ---
CURRENT_ALPHA=${ALPHAS[$TASK_ID]}

echo "Job array started on $(hostname) at $(date)"
echo "This is task $TASK_ID, running with alpha = $CURRENT_ALPHA"

# --- 模块加载 (不变) ---
nvidia-smi
module purge
module load python/3.11
module load gcc arrow/21.0.0
module load gcc opencv/4.12.0
module load scipy-stack/2024a
echo "Modules loaded."

source wildfire/bin/activate
echo "Virtual environment activated."

# --- 运行 (不变) ---
echo "Starting Python training script..."

# --- MODIFIED: Run the new script and pass the alpha as an argument ---
python -m src.ablation.ae_svcd_single $CURRENT_ALPHA

echo "Job finished at $(date)"