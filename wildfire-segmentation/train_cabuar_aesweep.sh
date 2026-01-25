#!/bin/bash 
#SBATCH --account=def-bereyhia
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --exclude=fc10512

# --- MODIFIED: Job Array ---
#SBATCH --array=0-15

# --- MODIFIED: Unique output log for each job ---
#SBATCH --output=slurm-%A_%a.out

# --- 1. 定义你的参数数组 ---
ALPHAS=(0.00001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.2 1.5 2 5)

# --- 2. 获取这个特定作业的索引 ---
TASK_ID=$SLURM_ARRAY_TASK_ID
CURRENT_ALPHA=${ALPHAS[$TASK_ID]}

echo "Job array started on $(hostname) at $(date)"
echo "This is task $TASK_ID, running with alpha = $CURRENT_ALPHA"

# --- 模块加载 (不变) ---
nvidia-smi
module purge
module load python/3.11
module load gcc arrow/17.0.0
module load scipy-stack/2024a
module load gcc opencv/4.12.0
echo "Modules loaded."

source wildfire/bin/activate
echo "Virtual environment activated."

# --- 4. 运行 Python 脚本 ---
echo "Running sweep for alpha = $CURRENT_ALPHA"
python -m src.ablation.cabuar_ae_sweep_single $CURRENT_ALPHA

echo "Job finished at $(date)"