#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# sweep の 1 試行を 1 GPU で実行する。
# 使い方:  qsub -g tga-hpc-lecture job.sh <ENTITY/kenkyu_project/SWEEP_ID>

source ~/.bash_profile
activate pytorch

# 計算ノードが外部ネットに出られない場合は offline にして、ジョブ後に `wandb sync` する。
# export WANDB_MODE=offline

wandb agent "$1"
