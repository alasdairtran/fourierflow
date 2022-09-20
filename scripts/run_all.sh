#!/bin/bash


# Super-resolution tests
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x64/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x64/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x64/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x64/256/config.yaml

CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x128/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x128/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x128/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x32_x128/256/config.yaml

CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64/256/config.yaml

CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64_x128/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64_x128/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64_x128/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_x64_x128/256/config.yaml
