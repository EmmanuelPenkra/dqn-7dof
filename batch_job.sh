Sample batch job:

#!/bin/bash

#SBATCH --account=advdls25
#SBATCH --job-name=dyt
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32  # CPUs
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1 # GPUs
#SBATCH --mem=128G
#SBATCH --time=02:00:00 # 8 hours
#SBATCH --output=ropeflash_%j.out
#SBATCH --error=ropeflash_%j.err

echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
nvidia-smi -L

# Load modules
module load arcc/1.0
module load gcc/13.2.0
module load cuda-toolkit/12.4.1

# Define variables for the interpreter path and number of GPUs
INTERPRETER_PATH=$(which python)
NUM_GPU=1

echo " testing..."
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d scannet -n semseg-pt-v3m1-ropeflash  -w model_best   

echo "All experiments completed."