#!/bin/bash
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l walltime=48:00:00
#PBS -l mem=96GB
#PBS -l jobfs=20MB
#PBS -P v89
#PBS -q gpuvolta
#PBS -l other=gdata1
#PBS -l storage=scratch/v89+gdata/v89
#PBS -M alasdair.tran@anu.edu.au
#PBS -N meshgraphnets-1
#PBS -j oe
#PBS -m abe
#PBS -l wd

source $HOME/.bashrc
cd /g/data/v89/at3219/projects/fourierflow/meshgraphnets

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/659/at3219/at3219/cuda-10.0/lib64

poetry run python run_model.py \
    --mode=train \
    --model=cfd \
    --checkpoint_dir=chk/cylinder_flow \
    --dataset_dir=../data/cylinder_flow
