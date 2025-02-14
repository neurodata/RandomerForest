#!/bin/bash

#SBATCH
#SBATCH --job-name=run_benchmarks
#SBATCH --array=0,1
#SBATCH --time=4-0:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=120000
#SBATCH --partition=parallel
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=tmtomita87@gmail.com

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

DATASETS=(adult connect_4)
D=${DATASETS[${SLURM_ARRAY_TASK_ID}]}

matlab -nosplash -nodisplay -singleCompThread -r "run_${D}_frc_2017_05_27;exit"

echo "job complete"
