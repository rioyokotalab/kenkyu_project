#!/bin/bash 
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# Pyenv
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# Modules
source /etc/profile.d/modules.sh
module load cuda cudnn openmpi nccl

# Run
mpirun -np 4 python 17_cifar10.py
