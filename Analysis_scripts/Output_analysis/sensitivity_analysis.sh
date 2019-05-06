#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l nodes=22:ppn=16
#PBS -j oe

cd $PBS_O_WORKDIR
source /etc/profile.d/modules.sh
module load python-2.7.5
mpirun python sensitivity_analysis.py
