# rivernet

Experiments with label propagation in networks of time series

## Getting Started

```sh
conda env create -f conda.yaml
conda activate rivernet
python -m ipykernel install --user --name rivernet --display-name "rivernet"
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
CUDA_VISIBLE_DEVICES=0 rivernet train configs/53_nbeats_vevo_full/config.yaml
CUDA_VISIBLE_DEVICES=0 rivernet train configs/54_vevo_perceiver/config.yaml
```
