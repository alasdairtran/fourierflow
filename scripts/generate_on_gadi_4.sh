#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l walltime=48:00:00
#PBS -l mem=384GB
#PBS -l jobfs=20MB
#PBS -P v89
#PBS -q gpuvolta
#PBS -l other=gdata1
#PBS -l storage=scratch/v89+gdata/v89
#PBS -M alasdair.tran@anu.edu.au
#PBS -N fourierflow-4
#PBS -j oe
#PBS -m abe
#PBS -l wd

source $HOME/.bashrc
cd /g/data/v89/at3219/projects/fourierflow

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run fourierflow generate --devices 0,1,2,3 --refresh 60 $DATA $CONFIG_PATH
