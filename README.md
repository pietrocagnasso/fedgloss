# Beyond Local Sharpness: Communication-Efficient Global Sharpness-aware Minimization for Federated Learning

This repository contains the official implementation of
> Caldarola, D., Cagnasso, P., Caputo, B., & Ciccone, M.
> [_Beyond Local Sharpness: Communication-Efficient Global Sharpness-aware Minimization for Federated Learning_](https://arxiv.org/abs/2412.03752),
> _IEEE Computer Vision and Pattern Recognition_ (CVPR) 2025.

## Abstract
Federated learning (FL) enables collaborative model training with privacy preservation. Data heterogeneity across edge devices (clients) can cause models to converge to sharp minima, negatively impacting generalization and robustness. Recent approaches use client-side sharpness-aware minimization (SAM) to encourage flatter minima, but the discrepancy between local and global loss landscapes often undermines their effectiveness, as optimizing for local sharpness does not ensure global flatness.  
This work introduces **FedGloSS** (**Fed**erated **Glo**bal **S**erver-side **S**harpness), a novel FL approach that prioritizes the optimization of global sharpness on the server, using SAM. To reduce communication overhead, FedGloSS cleverly approximates sharpness using the previous global gradient, eliminating the need for additional client communication. Our extensive evaluations demonstrate that FedGloSS consistently reaches flatter minima and better performance compared to state-of-the-art FL methods across various federated vision benchmarks.

<p align="center">
  <img src="fedgloss-vs-fedavg.gif" alt="GIF showing the comparison between FedGloSS and FedAVG" width="410" height="300px">
  <img src="fedgloss-vs-fedsam.gif" alt="GIF showing the comparison between FedGloSS and FedSAM" width="410" height="300px">
</p>

GIFs comparing FedGloSS loss landscape (net) against those of two well-known methods (solid), FedAVG on the left and FedSAM on the right. ResNet18 trained on CIFAR10 ($\alpha = 0.05$).

## Algorithm
**FedGloSS** proposes to apply SAM on the server side, while promoting consistency between local and global models by applying Alternating Direction Method of Multipliers (ADMM). Here, is the Algorithm that sumarizes the FedGloSS showing an example with both SAM and SGD (achievable by setting $\rho_l = 0$) on the client.

<p align="center">
  <img src="algorithm.png" alt="Algorithm summarizing FedGloSS", width="830px">
</p>

## Experiments

### Setup
#### Environment
Set up the environment and install the required dependencies:
```bash
python -m venv ../fgvenv
source ../fgvenv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

#### Datasets
After activating the environment, you can download and prepare the datasets using the provided script:
```bash
cd data
chmod +x setup_datasets.sh
./setup_datasets.sh
```

The provided datasets are CIFAR10 and CIFAR100. Both datasets are split equally among 100 clients, with varying degrees of heterogeneity controlled by Dirichlet's concentration parameter $\alpha$. The available splits are:

- CIFAR10: $\alpha \in \{0, 0.05, 1, 5, 10, 100\}$
- CIFAR100: $\alpha \in \{0, 0.5, 1000\}$


### Spectral analysis
The spectral analysis in `/models/eigenthings/` is based on the implementation from [`noahgolmant/pytorch-hessian-eigenthings`](https://github.com/noahgolmant/pytorch-hessian-eigenthings).

### Run the experiments in pytorch
Below are examples of how to run experiments as described in the paper. The heterogeneity level can be adjusted using the `--dir-alpha` parameter.

#### CIFAR10
```bash
python main.py --dataset cifar10 --dir-alpha 0 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --seed 0 --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000
```

#### CIFAR100
```bash
python main.py --dataset cifar100 --dir-alpha 0 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --seed 0 --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000
```

### Results
The execution generates the following output files:
- `./models/results/[YYYY][mm][dd]_[HH]:[MM]:[SS]/params.csv`: Summary of the parameters used for the run.
- `./models/results/[YYYY][mm][dd]_[HH]:[MM]:[SS]/trends.csv`: Results in terms of accuracy, global model norm, and pseudo-gradient norm.
- `./models/results/[YYYY][mm][dd]_[HH]:[MM]:[SS]/eigs.csv`: First five eigenvalues and the ratio between the first and fifth eigenvalue.
