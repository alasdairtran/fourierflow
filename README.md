# fourierflow

Experiments with Fourier layers on simulation data.

## Getting Started

```sh
# Set up pyenv and pin python version to 3.9.6
curl https://pyenv.run | bash
# Configure our shell's environment for pyenv
pyenv install 3.9.6
pyenv local 3.9.6

# Set up poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
export PATH="$HOME/.poetry/bin:$PATH"

# Install all python dependencies
poetry install
source .venv/bin/activate # or: poetry shell
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
# Manually install tensorflow as poetry gets mad when tensorflow wants newer
# tensorboard version than what pytorch-lightning requires.
poe install-tensorflow
# Manually reinstall Pytorch with CUDA 11.1 support
poe install-torch-cuda11

# set default paths
cp example.env .env
# The environmnet variables in .env will be loaded automatically when running
# fourierflow train, but we can also load them manually in our terminal
export $(cat .env | xargs)

# Alternatively, you can pass the paths to the system using env vars, e.g.
FNO_DATA_ROOT=/My/Data/Location fourierflow
```

## Navier Stokes Experiments

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
# take 10 times as long, while the differnece between the solutions in step 20
# is only 0.04%.

# Reproducing SOA model on Navier Stokes.
fourierflow train configs/navier_stokes_4/01_li_baseline/config.yaml

# Train with our best model
fourierflow train configs/navier_stokes_4/02_best_200/config.yaml

# Performance tradeoff evaluation. We use the Navier Stokes test set
# as our benchmark dataset.

# Train on gadi
qsub -N 02_best_200 -v CONFIG=configs/navier_stokes_4/02_best_200/config.yaml /g/data/v89/at3219/projects/fourierflow/scripts/start_gadi_job.sh
```

## Meshgraphnet Experiments

```sh
# DeepMind meshgraphnets simulation data
cd meshgraphnets
sh download_dataset.sh airfoil data
sh download_dataset.sh cylinder_flow data
sh download_dataset.sh deforming_plate data
sh download_dataset.sh flag_minimal data
sh download_dataset.sh flag_simple data
sh download_dataset.sh flag_dynamic data
sh download_dataset.sh flag_dynamic_sizing data
sh download_dataset.sh sphere_simple data
sh download_dataset.sh sphere_dynamic data
sh download_dataset.sh sphere_dynamic_sizing data

# Convert to HDF5 format
python scripts/convert.py cylinder-flow

# Create index files
python -m tfrecord.tools.tfrecord2idx data/cylinder_flow/train.tfrecord data/cylinder_flow/train.index
python -m tfrecord.tools.tfrecord2idx data/cylinder_flow/valid.tfrecord data/cylinder_flow/valid.index
python -m tfrecord.tools.tfrecord2idx data/cylinder_flow/test.tfrecord data/cylinder_flow/test.index

# Reproduce mesh experiment
pyenv local 3.6.14
virtualenv .venv/mesh
source .venv/mesh/bin/activate
pip install -r requirements.txt
# Set LD_LIBRARY_PATH to cuda-10.0/lib64 location

# Train
python run_model.py --mode=train --model=cloth --checkpoint_dir=chk/flag_simple --dataset_dir=data/flag_simple
python run_model.py --mode=train --model=cfd --checkpoint_dir=chk/cylinder_flow --dataset_dir=data/cylinder_flow
# Roll out
python run_model.py --mode=eval --model=cloth --checkpoint_dir=chk/flag_simple --dataset_dir=data/flag_simple --rollout_path=data/flag_simple/rollout_flag.pkl
python run_model.py --mode=eval --model=cfd --checkpoint_dir=chk/cylinder_flow --dataset_dir=data/cylinder_flow --rollout_path=data/cylinder_flow/rollout_flag.pkl
# Plot
python plot_cloth.py --rollout_path=data/flag_simple/rollout_flag.pkl
python plot_cfd.py --rollout_path=data/cylinder_flow/rollout_flag.pkl
```

# Time Series Experiments

```sh
# MIMIC-III dataset
cd data/mimiciii/1.4 && gzip -d *gz

# Vevo experiments
fourierflow train configs/vevo/02_nbeats/config.yaml
fourierflow train configs/vevo/03_radflow/config.yaml
# N-BEATS with fourier layer - similar performance
fourierflow train configs/54_vevo_perceiver/config.yaml
```
