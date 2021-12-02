![Teaser](https://raw.githubusercontent.com/alasdairtran/fourierflow/main/figures/poster.png)

# Factorized Fourier Neural Operators

This repository contains the code to reproduce the results in our [NeurIPS 2021
ML4PS workshop](https://ml4physicalsciences.github.io/2021/) paper, [Factorized
Fourier Neural Operators](https://arxiv.org/abs/2111.13802).

The Fourier Neural Operator (FNO) is a learning-based method for efficiently
simulating partial differential equations. We propose the Factorized Fourier
Neural Operator (F-FNO) that allows much better generalization with deeper
networks. With a careful combination of the Fourier factorization, weight
sharing, the Markov property, and residual connections, F-FNOs achieve a
six-fold reduction in error on the most turbulent setting of the Navier-Stokes
benchmark dataset. We show that our model maintains an error rate of 2% while
still running an order of magnitude faster than a numerical solver, even when
the problem setting is extended to include additional contexts such as
viscosity and time-varying forces. This enables the same pretrained neural
network to model vastly different conditions.

Please cite with the following BibTeX:

```raw
@misc{Tran2021Factorized,
  title         = {Factorized Fourier Neural Operators},
  author        = {Alasdair Tran and Alexander Mathews and Lexing Xie and Cheng Soon Ong},
  year          = {2021},
  eprint        = {2111.13802},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}
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
pyenv install 3.9.9
pyenv local 3.9.9

# Set up poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Install all python dependencies
poetry install
source .venv/bin/activate # or: poetry shell
# If we need to use Jupyter notebooks
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
# Provide a more meaningful prompt when inside the virtual environment
python -m venv .venv --prompt fourierflow
# Manually reinstall Pytorch with CUDA 11.3 support
# Monitor poetry's support for pytorch here: https://github.com/python-poetry/poetry/issues/2613
poe install-torch-cuda11

# set default paths
cp example.env .env
# The environment variables in .env will be loaded automatically when running
# fourierflow train, but we can also load them manually in our terminal
export $(cat .env | xargs)

# Alternatively, you can pass the paths to the system using env vars, e.g.
FNO_DATA_ROOT=/My/Data/Location fourierflow
```

## Navier Stokes Experiments

You can download all of our datasets and pretrained model as follows:

```sh
# Datasets (209GB)
wget --continue https://object-store.rc.nectar.org.au/v1/AUTH_c0e4d64401cf433fb0260d211c3f23f8/fourierflow/data.tar.gz
tar -zxvf data.tar.gz

# Pretrained models and results (30GB)
wget --continue https://object-store.rc.nectar.org.au/v1/AUTH_c0e4d64401cf433fb0260d211c3f23f8/fourierflow/experiments.tar.gz
tar -zxvf experiments.tar.gz
```

Alternatively, you can also generate the datasets from scratch:

```sh
# Download Navier Stokes datasets
fourierflow download fno

# Generate Navier Stokes on toruses with a different forcing function and
# viscosity for each sample. Takes 14 hours.
fourierflow generate navier-stokes --force random --cycles 2 --mu-min 1e-5 \
    --mu-max 1e-4 --steps 200 --delta 1e-4 \
    data/navier-stokes/random_force_mu.h5

# Generate Navier Stokes on toruses with a different time-varying forcing
# function and a different viscosity for each sample. Takes 21 hours.
fourierflow generate navier-stokes --force random --cycles 2 --mu-min 1e-5 \
    --mu-max 1e-4 --steps 200 --delta 1e-4 --varying-force \
    data/navier-stokes/random_varying_force_mu.h5

# If we decrease delta from 1e-4 to 1e-5, generating the same dataset would now
# take 10 times as long, while the difference between the solutions in step 20
# is only 0.04%.
```

Training and test commands:

```sh
# Reproducing SOA model on Navier Stokes from Li et al (2021).
fourierflow train --trial 0 experiments/navier_stokes_4/zongyi/4_layers

# Train with our best model
fourierflow train --trial 0 experiments/navier_stokes_4/markov/24_layers

# Get inference time on test set
fourierflow predict --trial 0 experiments/navier_stokes_4/markov/24_layers
```

Visualization commands:

```sh
# Create all plots and tables for paper
fourierflow plot layer
fourierflow plot complexity
fourierflow plot table-3

# Create the flow animation for presentation
fourierflow plot flow

# Create plots for the poster
fourierflow plot poster
```

## Acknowledgement

Our model is based on the code of the original author of the Fourier Neural
Operators paper: https://github.com/zongyi-li/fourier_neural_operator
