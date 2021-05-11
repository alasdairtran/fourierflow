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
source ./venv/bin/activate
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
# Manually reinstall Pytorch with CUDA 11.1 support
pip install -U torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

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
