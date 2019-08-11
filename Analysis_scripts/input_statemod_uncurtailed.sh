#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=42             # specify number of nodes
#SBATCH --ntasks-per-node=24  # specify number of core per node
#SBATCH --export=ALL
#SBATCH -t 0:45:00            # set max wallclock time
#SBATCH --job-name="statemod" # name your job 
#SBATCH --output="output/output.out"

module load python
module load mpi4py
module load scipy
ibrun python input_statemod_uncurtailed.py
