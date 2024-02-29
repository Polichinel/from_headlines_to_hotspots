#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-551 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-0:50:00

# Read the given line from the input file and evaluate it:
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 SciPy-bundle/2022.05-foss-2022a  IPython scikit-learn/1.1.2-foss-2022a matplotlib/3.5.2-foss-2022a
source /cephyr/users/croicu/Alvis/marisol/bin/activate

echo 5486
python -u predict.py --seed=5486 --shard_id=${SLURM_ARRAY_TASK_ID}

echo 34688
python -u predict.py --seed=34688 --shard_id=${SLURM_ARRAY_TASK_ID}

echo 12958
python -u predict.py --seed=12958 --shard_id=${SLURM_ARRAY_TASK_ID}

echo 11268
python -u predict.py --seed=11268 --shard_id=${SLURM_ARRAY_TASK_ID}

echo 73180
python -u predict.py --seed=73180 --shard_id=${SLURM_ARRAY_TASK_ID}
