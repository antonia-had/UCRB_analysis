#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=15 
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 1:30:00
#SBATCH --job-name="LR"
#SBATCH --output="../output/outputLR.out"

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0y
ibrun python3 LR_factor_mapping.py LHsamples_original_1000