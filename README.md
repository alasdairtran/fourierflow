# fourierflow

Experiments with Fourier layers on simulation data.

## Getting Started

```sh
# Set up pyenv and pin python version to 3.9.5
curl https://pyenv.run | bash
# Configure our shell's environment for pyenv
pyenv install 3.9.5
pyenv local 3.9.5

# Set up poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
export PATH="$HOME/.poetry/bin:$PATH"

# Install all python dependencies
poetry install
source .venv/bin/activate # or: poetry shell
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
# Manually reinstall Pytorch with CUDA 11.1 support
poe install-torch-cuda11

# set default paths
cp example.env .env

# Alternatively, you can pass the paths to the system using env vars, e.g. 
FNO_DATA_ROOT=/My/Data/Location fourierflow 

# Download Navier Stokes datasets
fourierflow download-fno-examples

# MIMIC-III dataset
cd data/mimiciii/1.4 && gzip -d *gz

# DeepMind meshgraphnets simulation data
cd deepmind-research/meshgraphnets
sh download_dataset.sh airfoil data
sh download_dataset.sh cylinder_flow data
sh download_dataset.sh deforming_plate data
sh download_dataset.sh flag_minimal data
sh download_dataset.sh flag_simple data
sh download_dataset.sh flag_dynamic data
sh download_dataset.sh sphere_simple data
sh download_dataset.sh sphere_dynamic data
# Reproduce mesh experiment
virtualenv --python=python3.6 .venv/mesh
source .venv/mesh/bin/activate
pip install -r requirements.txt
# Train
python -m meshgraphnets.run_model --mode=train --model=cfd \
    --checkpoint_dir=data/chk --dataset_dir=data/flag_simple
```

## Training

```sh
# Vevo experiments
fourierflow train configs/vevo/02_nbeats/config.yaml
fourierflow train configs/vevo/03_radflow/config.yaml
# N-BEATS with fourier layer - similar performance
fourierflow train configs/54_vevo_perceiver/config.yaml

# Reproducing SOA model on Navier Stokes
fourierflow train configs/navier_stokes_4/01_li_baseline/config.yaml

# With Radflow
fourierflow train configs/navier_stokes_4p/111b_single_24_40e/config.yaml
```

## Evaluation

```sh
# Performance tradeoff evaluation. We use the Navier Stokes test set
# as our benchmark dataset.
```
