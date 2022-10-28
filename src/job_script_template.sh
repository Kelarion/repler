#!/bin/bash
#SBATCH --job-name=bigjob
#SBATCH --account=theory
#SBATCH -c 8
#SBATCH --time=0-11:59:00
#SBATCH --mem-per-cpu=6gb
#SBATCH --array=0-{n_tot}%50
module load anaconda/3-2020.11
source activate pete
python {file_dir}/run_experiment.py $SLURM_ARRAY_TASK_ID {n_task}
# python -c "print('success')"