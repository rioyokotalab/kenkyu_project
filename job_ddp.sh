#!/bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# 1 ノード 4 GPU で DDP 学習する。node_f=1 = F ノード 1 台 = H100 4 枚（node_q=1 は 1/4 ノード = GPU 1 枚）。
# 使い方:  qsub -g tga-hpc-lecture job_ddp.sh

source ~/.bash_profile
activate pytorch
export PYTHONUNBUFFERED=1   # print をすぐ出力に出す

# export WANDB_MODE=offline   # 計算ノードが外部ネット不可のとき

torchrun --standalone --nproc_per_node=4 14_ddp.py --tokenizer char --block_size 256 --n_layer 6 --n_head 6 --n_embd 384 --max_iters 3000 --batch_size 32
