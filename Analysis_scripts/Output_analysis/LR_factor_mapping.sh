#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1             # specify number of nodes
#SBATCH --ntasks-per-node=10  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 0:30:00            # set max wallclock time
#SBATCH --job-name="factormapping" # name your job
#SBATCH --output="output.out"

module load python
module load mpi4py
module load scipy
ibrun python LR_factor_mapping.py
