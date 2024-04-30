#!/bin/bash 
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o output/o.$JOB_ID

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

module load nvhpc nccl
wandb agent $1
