#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 3:30:00
#SBATCH --job-name="LR"
#SBATCH --output="outputLR.out"

module load python
module load mpi4py
module load scipy
ibrun python LR_factor_mapping.py