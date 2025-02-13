#!/bin/bash
#SBATCH --job-name=VAE               
#SBATCH --output=vaeformer            
#SBATCH --partition=ai4earth          
#SBATCH --quotatype=reserved
#SBATCH --ntasks-per-node=8           
#SBATCH --cpus-per-task=8              
#SBATCH --gres=gpu:8                  
#SBATCH --nodes=1
#SBATCH --mem=64G                 # 内存需求
#SBATCH --time=480:00:00          # 运行时间（1小时）

source ~/.bashrc
conda activate FNP

echo "SLURM_STEP_NODELIST: $SLURM_STEP_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_SRUN_COMM_PORT: $SLURM_SRUN_COMM_PORT"


srun python -u train.py --world_size=8 --per_cpus=4 --batch_size=2

