#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=8             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 1:00:00            # set max wallclock time
#SBATCH --job-name="streamflow" # name your job 
#SBATCH --output="../output/streamflow.out"

module load python
module load mpi4py
module load scipy
ibrun python 15mile_streamflow.py LHsamples_original_1000