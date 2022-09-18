#!/bin/bash


# Super-resolution tests
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64/256/config.yaml

CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64_128/32/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64_128/64/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64_128/128/config.yaml
CUDA_VISIBLE_DEVICES=0 fourierflow test --trial 0 --force experiments/torus_kochkov/ffno/superresolution/train_with_64_128/256/config.yaml
