# fourierflow

Experiments with label propagation in networks of time series

## Getting Started

```sh
# Set up pyenv and pin python version to 3.8.9
brew install pyenv
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
source ~/.zshrc
pyenv install 3.8.9
pyenv local 3.8.9
# Set up poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
echo -e 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
# Install all python dependencies
poetry install
source .venv/bin/activate
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
# Manually reinstall Pytorch with CUDA 11.1 support
pip install -U torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/google-research/torchsde.git

# Download Navier Stokes datasets
mkdir data/fourier && cd data/fourier
gdown --id 16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe # Burgers_R10.zip
gdown --id 1nzT0-Tu-LS2SoMUCcmO1qyjQd6WC9OdJ # Burgers_v100.zip
gdown --id 1G9IW_2shmfgprPYISYt_YS8xa87p4atu # Burgers_v1000.zip
gdown --id 1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV # Darcy_241.zip
gdown --id 1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf # Darcy_421.zip
gdown --id 1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d # NavierStokes_V1e-3_N5000_T50.zip
gdown --id 1pr_Up54tNADCGhF8WLvmyTfKlCD5eEkI # NavierStokes_V1e-4_N20_T50_R256_test.zip
gdown --id 1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3 # NavierStokes_V1e-4_N10000_T30.zip
gdown --id 1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5 # NavierStokes_V1e-5_N1200_T20.zip
unzip *.zip && rm -rf *.zip

# MIMIC-III dataset
cd data/mimiciii/1.4 && gzip -d *gz
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
```
