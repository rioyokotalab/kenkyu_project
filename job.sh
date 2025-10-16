#!/bin/bash 
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o output/o.$JOB_ID

source ~/.bash_profile
activate pytorch
wandb agent $1
