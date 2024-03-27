#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-551 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 2-00:00:00

echo "Running desloth prediction"

module load SciPy-bundle/2022.05-foss-2022a wandb/0.13.4 scikit-learn/1.1.2-foss-2022a 
source /cephyr/users/croicu/Alvis/marisol/bin/activate

python -u tps2.py


