#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=15 
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 0:30:00
#SBATCH --job-name="LR"
#SBATCH --output="../output/outputLR.out"

module load python
module load mpi4py
module load scipy
ibrun python LR_factor_mapping.py LHsamples_original_1000