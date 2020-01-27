#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=32             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 12:00:00            # set max wallclock time
#SBATCH --job-name="infofiles" # name your job 
#SBATCH --output="../output/infofiles.out"
#SBATCH --mail-user=ah986@cornell.edu
#SBATCH --mail-type=ALL

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0
ibrun python3 infofile_uncurtailed.py Sobol_sample
