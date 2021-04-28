# fourierflow

Experiments with label propagation in networks of time series

## Getting Started

```sh
conda env create -f conda.yaml
conda activate fourierflow
python -m ipykernel install --user --name fourierflow --display-name "fourierflow"
python setup.py develop

# Ensure PyTorch Geometric is compatible with CUDA and Pytorch versions
export TORCH_VERSION=1.7.0
export CUDA_VERSION=cu110
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install -U torch-geometric
```

## Training

```sh
# N-BEATS baseline
fourierflow train configs/53_nbeats_vevo_full/config.yaml
# N-BEATS with fourier layer - similar performance
fourierflow train configs/54_vevo_perceiver/config.yaml

# Reproducing SOA model on Navier Stokes
fourierflow train configs/41_navier_stokes_2d_baseline/config.yaml
```
