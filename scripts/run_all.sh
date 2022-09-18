#!/bin/bash


# Super-resolution tests
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx,ky,lap experiments/torus_kochkov/ffno/superresolution/train_with_64/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx,ky,lap experiments/torus_kochkov/ffno/superresolution/train_with_64/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx,ky,lap experiments/torus_kochkov/ffno/superresolution/train_with_64/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx,ky,lap experiments/torus_kochkov/ffno/superresolution/train_with_64/256/config.yaml

CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx_64,ky_64,lap_64,kx_128,ky_128,lap_128 experiments/torus_kochkov/ffno/superresolution/train_with_64_128/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx_128,ky_128,lap_128 experiments/torus_kochkov/ffno/superresolution/train_with_64_128/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx_64,ky_64,lap_64 experiments/torus_kochkov/ffno/superresolution/train_with_64_128/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force --remove-keys kx_64,ky_64,lap_64,kx_128,ky_128,lap_128 experiments/torus_kochkov/ffno/superresolution/train_with_64_128/256/config.yaml
