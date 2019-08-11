#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 0:30:00
#SBATCH --job-name="LR"
#SBATCH --output="output/outputLR.out"

module load python
module load mpi4py
module load scipy
ibrun python LR_factor_mapping.py