#!/bin/bash -l

#SBATCH
#SBATCH --job-name=UCI_Datarun
#SBATCH --time=6:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=shared
#SBATCH --mail-type=end
#SBATCH --mail-user=jpatsol1@jhu.edu


##
## This is used for setup between MARCC and dev on my local box.
## 

if [[ "$USER" == "jpatsol1@jhu.edu" ]]; then
	export NCORES=$SLURM_NTASKS
	module load gcc/6.4.0
	module load python/3.7
	source ./env/bin/activate
	module load gcc/6.4.0
elif [[ "$USER" == "JLP" ]]; then
	#export SLURM_ARRAY_TASK_ID=117 #wine rerf
	#export SLURM_ARRAY_TASK_ID=238 #wine RF
	#export SLURM_ARRAY_TASK_ID=359 #wine SKRF
	#export SLURM_ARRAY_TASK_ID=480 #wine SKX
	export SLURM_ARRAY_TASK_ID=2 #abalone rerf
	export NCORES=1
fi


##
## The file `jobMap.dat` contains the parameters for each run
## 
DATASET="$(awk '{print $1}' jobMap_filtered_numeric.dat | sed "$SLURM_ARRAY_TASK_ID q;d")"
CLASSIFIER="$(awk '{print $2}' jobMap_filtered_numeric.dat | sed "$SLURM_ARRAY_TASK_ID q;d")"

echo "DATASET=$DATASET	CLASSIFIER=$CLASSIFIER"
## Run
python run.py $DATASET $CLASSIFIER


#################### 
#################### 
### Instructions ###
#################### 
#################### 

## To run on MARCC:
# the file jobMap.dat file has 484 lines.
# the file jobMap_filtered_numeric.dat file has 220 lines.
# So, we'll submit an array job (50 jobs max at once).
# The environment variable SLURM_ARRAY_TASK_ID will enable 
# reading of parameters from the jobMap.dat file.
#
# sbatch -o SlurmOUT/slurm-%A_%3a.out --array=1-220%50 master.scr

