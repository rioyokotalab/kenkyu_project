#!/bin/bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# 1 ノード 4 GPU で DDP 学習する（node_q = F ノード 1 台 = 4 GPU）。
# 使い方:  qsub -g tga-hpc-lecture job_ddp.sh

source ~/.bash_profile
activate pytorch

# export WANDB_MODE=offline   # 計算ノードが外部ネット不可のとき

torchrun --standalone --nproc_per_node=4 14_ddp.py --max_iters 5000 --batch_size 32
