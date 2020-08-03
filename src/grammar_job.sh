#!/bin/sh
#SBATCH --account=theory
#SBATCH --job-name=geom
#SBATCH -c 1
#SBATCH --time=1-11:59:00
#SBATCH --mem-per-cpu=4gb
#SBATCH --array=[0-29]%10
module load anaconda/3-5.1
source activate pete
python repler/grammar_script.py $SLURM_ARRAY_TASK_ID   # Run the job steps
