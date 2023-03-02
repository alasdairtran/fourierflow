![Teaser](https://raw.githubusercontent.com/alasdairtran/fourierflow/main/figures/poster.png)

# Factorized Fourier Neural Operators

This repository contains the code to reproduce the results in our [ICLR
2023](https://iclr.cc/Conferences/2023) paper, [Factorized Fourier Neural
Operators](https://arxiv.org/abs/2111.13802).

We propose the Factorized Fourier Neural Operator (F-FNO), a learning-based
approach for simulating partial differential equations (PDEs). Starting from a
recently proposed Fourier representation of flow fields, the F-FNO bridges the
performance gap between pure machine learning approaches to that of the best
numerical or hybrid solvers. This is achieved with new representations –
separable spectral layers and improved residual connections – and a combination
of training strategies such as the Markov assumption, Gaussian noise, and
cosine learning rate decay. On several challenging benchmark PDEs on regular
grids, structured meshes, and point clouds, the F-FNO can scale to deeper
networks and outperform both the FNO and the geo-FNO, reducing the error by 83%
on the Navier-Stokes problem, 31% on the elasticity problem, 57% on the airfoil
flow problem, and 60% on the plastic forging problem. Compared to the
state-of-the-art pseudo-spectral method, the F-FNO can take a step size that is
an order of magnitude larger in time and achieve an order of magnitude speedup
to produce the same solution quality.

Please cite with the following BibTeX:

```raw
@inproceedings{tran2023factorized,
title={Factorized Fourier Neural Operators},
author={Alasdair Tran and Alexander Mathews and Lexing Xie and Cheng Soon Ong},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=tmIiMPl4IPa}
}
```

## Getting Started

```sh
# If you'd like to reproduce the exact numbers in our ML4PS workshop paper,
# checkout v0.2.0 to use the same seeds and package versions.
git checkout v0.2.0

# Otherwise we can use latest Python and Pytorch versions.
# Set up pyenv and pin python version to 3.9.9
curl https://pyenv.run | bash
# Configure our shell's environment for pyenv
pyenv install 3.10.7
pyenv local 3.10.7
curl -sSL https://install.python-poetry.org | python3 - --version 1.2.0b3

# Install all python dependencies
poetry install
source .venv/bin/activate # or: poetry shell
# If we need to use Jupyter notebooks
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"

# set default paths
cp example.env .env
# The environment variables in .env will be loaded automatically when running
# fourierflow train, but we can also load them manually in our terminal
export $(cat .env | xargs)

# Alternatively, you can pass the paths to the system using env vars, e.g.
DATA_ROOT=/My/Data/Location fourierflow train ...
```

## Navier Stokes Experiments

You can download all of our datasets and pretrained model as follows:

```sh
# Datasets (209GB)
wget --continue https://object-store.rc.nectar.org.au/v1/AUTH_c0e4d64401cf433fb0260d211c3f23f8/fourierflow/data-2021-12-24.tar.gz
tar -zxvf data-2021-12-24.tar.gz

# Pretrained models and results (30GB)
wget --continue https://object-store.rc.nectar.org.au/v1/AUTH_c0e4d64401cf433fb0260d211c3f23f8/fourierflow/experiments-2021-12-24.tar.gz
tar -zxvf experiments-2021-12-24.tar.gz
```

Alternatively, you can also generate the datasets from scratch:

```sh
# Download Navier Stokes datasets
fourierflow download fno

# Generate Navier Stokes on toruses with a different forcing function and
# viscosity for each sample. Takes 14 hours.
fourierflow generate navier-stokes --force random --cycles 2 --mu-min 1e-5 \
    --mu-max 1e-4 --steps 200 --delta 1e-4 \
    data/torus/torus_vis.h5

# Generate Navier Stokes on toruses with a different time-varying forcing
# function and a different viscosity for each sample. Takes 21 hours.
fourierflow generate navier-stokes --force random --cycles 2 --mu-min 1e-5 \
    --mu-max 1e-4 --steps 200 --delta 1e-4 --varying-force \
    data/torus/torus_vis_force.h5

# If we decrease delta from 1e-4 to 1e-5, generating the same dataset would now
# take 10 times as long, while the difference between the solutions in step 20
# is only 0.04%.

# Generate initial conditions for 2D Kolmogorov flows (Kochkov et al, 2021).
fourierflow generate kolmogorov data/kolmogorov/re_1000/initial_conditions/train.yaml # 22 GPU hours
fourierflow generate kolmogorov data/kolmogorov/re_1000/initial_conditions/valid.yaml # 3 GPU hours
fourierflow generate kolmogorov data/kolmogorov/re_1000/initial_conditions/test.yaml # 22 GPU hours

# Run baseline simulations with the numerical solver (no ML).
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/32.yaml # 1 GPU min
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/64.yaml # 2 GPU mins
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/128.yaml # 3 GPU mins
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/256.yaml # 6 GPU mins
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/512.yaml # 20 GPU mins
fourierflow generate kolmogorov data/kolmogorov/re_1000/baseline/1024.yaml # 2 GPU hours

# Generating training data for ML models.
fourierflow generate kolmogorov data/kolmogorov/re_1000/trajectories/train.yaml # 19 GPU hours
fourierflow generate kolmogorov data/kolmogorov/re_1000/trajectories/valid.yaml # 2 GPU hours
fourierflow generate kolmogorov data/kolmogorov/re_1000/trajectories/test.yaml # 19 GPU hours
```

Training and test commands:

```sh
# Reproducing SOA model on Navier Stokes from Li et al (2021).
fourierflow train --trial 0 experiments/torus_li/zongyi/4_layers/config.yaml

# Train with our best model
fourierflow train --trial 0 experiments/torus_li/markov/24_layers/config.yaml

# Get inference time on test set
fourierflow predict --trial 0 experiments/torus_li/markov/24_layers/config.yaml
```

Visualization commands:

```sh
# Some example commands to create plots and tables for paper
fourierflow plot torus-li-performance
fourierflow plot complexity
fourierflow plot table-torus-li
fourierflow plot table-airfoil
fourierflow plot table-elasticity
fourierflow plot table-plasticity

# Create the flow animation for presentation
fourierflow plot flow

# Create plots for the poster
fourierflow plot poster

# Create plots related to comparison with 2D Kolmogorov flows (Kochkov et al, 2021).
fourierflow plot correlation
```

## Experiments with JAX-CFD

```sh
# Download jax-cfd datasets
gsutil -m cp -r gs://gresearch/jax-cfd data/
```

## Plots

```sh
# Plot effect of coordinates and velocity as input channels (Figure 5b)
fourierflow plot coordinates-velocity-ablation
```

## Geo-FNO experiments

```sh
# Download the Geo-FNO datasets
fourierflow download geo-fno

# Reproducing SOA model from Li et al (2022).
fourierflow train --trial 0 experiments/airfoil/geo-fno/4_layers/config.yaml
fourierflow train --trial 0 experiments/plasticity/geo-fno/4_layers/config.yaml
fourierflow train --trial 0 experiments/elasticity/geo-fno/4_layers/config.yaml

# Train with our best model
fourierflow train --trial 0 experiments/airfoil/ffno/24_layers/config.yaml
fourierflow train --trial 0 experiments/plasticity/ffno/24_layers/config.yaml
fourierflow train --trial 0 experiments/elasticity/ffno/24_layers/config.yaml

# Plot samples
fourierflow sample experiments/elasticity/geo-fno/4_layers/config.yaml
fourierflow sample experiments/elasticity/ffno/24_layers/config.yaml

fourierflow sample experiments/elasticity/ffno/24_layers/config.yaml
```

<!-- ## Mesh Experiments

```sh
# DeepMind meshgraphnets simulation data
fourierflow download meshgraphnets
# Convert cylinder-flow data from TFRecords to HDF5 format.
fourierflow convert cylinder-flow --data-dir data/meshgraphnets/cylinder_flow --out data/meshgraphnets/cylinder_flow/cylinder_flow.h5
``` -->

## Acknowledgement

Our model is based on the code of the original author of the Fourier Neural
Operators paper:

* https://github.com/zongyi-li/fourier_neural_operator
* https://github.com/zongyi-li/Geo-FNO

JAX-based models are adapted from JAX-CFD: https://github.com/google/jax-cfd

Mesh-based simulations are based on meshgraphnets: https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
