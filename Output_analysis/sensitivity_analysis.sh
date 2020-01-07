#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=8             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 5:00:00            # set max wallclock time
#SBATCH --job-name="sensitivityanalysis" # name your job
#SBATCH --output="../output/sensitivityanalysis.out"

module load python
module load scipy/3.6
export MODULEPATH=/share/apps/compute/modulefiles/applications:$MODULEPATH
module load mpi4py
export MV2_ENABLE_AFFINITY=0
ibrun python3 sensitivity_analysis.py LHsamples_original_1000
