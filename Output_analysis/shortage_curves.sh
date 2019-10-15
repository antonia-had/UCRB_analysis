#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 0:30:00            # set max wallclock time
#SBATCH --job-name="curves" # name your job
#SBATCH --output="output/curves.out"

module load python
module load mpi4py
module load scipy
ibrun python shortage_duration_curves_global_experiment.py
