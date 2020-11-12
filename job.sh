#!/bin/bash 
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=23:00:00
#$ -j y
#$ -o output/o.$JOB_ID

# Run
source ~/.bash_profile
$1
