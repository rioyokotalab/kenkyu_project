#!/bin/bash 
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -o output/o.$JOB_ID

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

source /etc/profile.d/modules.sh
module load cuda openmpi nccl cudnn
$1
