#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=8             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 1:00:00            # set max wallclock time
#SBATCH --job-name="sensitivityanalysis" # name your job
#SBATCH --output="../output/sensitivityanalysis.out"

module load python
module load mpi4py
module load scipy
ibrun python sensitivity_analysis.py
