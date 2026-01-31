<p align="center">
  <img src="/images/logo_well_spread.png" style="width: 30%; height: auto;">
</p>
<!--
[![PyPI version](https://badge.fury.io/py/moospread.svg)](https://badge.fury.io/py/moospread)
-->
<p align="center">
  <a href="https://pypi.org/project/moospread/"><img src="https://img.shields.io/pypi/v/moospread.svg" alt="PyPI version"></a>
  <a href="https://moospread.readthedocs.io">
  <img src="https://img.shields.io/badge/docs-online-brightgreen.svg" alt="Documentation">
</a>
</p>
<div align="center">
    <h3>
      <a href="https://pypi.org/project/moospread/">Installation</a> |
      <a href="https://moospread.readthedocs.io/en/latest/">Documentation</a> | 
      <a href="https://arxiv.org/pdf/2509.21058">Paper</a>
    </h3>
</div>
<!--
<a href="https://pypi.org/project/moospread/"><img src="https://img.shields.io/pypi/v/advermorel.svg" alt="PyPI version"></a>
-->

# [ICLR 2026] SPREAD: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion

> SPREAD is a novel sampling-based approach for multi-objective optimization that leverages diffusion models to efficiently refine and generate well-spread Pareto front approximations. It combines the expressiveness of diffusion models with multi-objective optimization principles to achieve both high convergence to the Pareto front and excellent diversity across the objective space. SPREAD demonstrates competitive performance against state-of-the-art methods while providing a flexible framework for different optimization contexts.

## 🚀 Getting Started

### Installation

```python
conda create -n moospread python=3.11
conda activate moospread
pip install moospread
```
Or, to install the latest code from GitHub:
```python
conda create -n moospread python=3.11
conda activate moospread
git clone https://github.com/safe-autonomous-systems/moo-spread.git
cd moo-spread
pip install -e .
```
### Basic usage
This example shows how to solve a standard multi-objective optimization benchmark (ZDT2) using the **SPREAD** solver.

```python
import numpy as np
import torch

# Import the SPREAD solver
from moospread import SPREAD

# Import a test problem
from moospread.tasks import ZDT2

# Define the problem
problem = ZDT2(n_var=30)

# Initialize the SPREAD solver
solver = SPREAD(
    problem,
    data_size=10000,
    timesteps=1000,
    num_epochs=1000,
    train_tol=100,
    mode="online",
    seed=2026,
    verbose=True
)

# Solve the problem
res_x, res_y = solver.solve(
    num_points_sample=200,
    iterative_plot=True,
    plot_period=10,
    max_backtracks=25,
    save_results=True,
    samples_store_path="./samples_dir/",
    images_store_path="./images_dir/"
)
```

This will train a diffusion-based multi-objective solver, approximate the Pareto front of the ZDT2 problem, and store generated samples and plots in the specified directories.

---


### 📚 Next steps

For more advanced examples (offline mode, mobo mode, tutorials), see the full [documentation](https://moospread.readthedocs.io/en/latest/).

## 🔬 Experiments

All experiment code is contained in the `/experiments` directory:

* **Online setting:** `/experiments/spread/`
* **Offline setting:** `/experiments/spread_offline/`
* **Bayesian setting:** `/experiments/spread_bayesian/`

The following Jupyter notebooks reproduce the plots shown in our paper:

* `/experiments/spread/notebook_online_spread.ipynb`
* `/experiments/spread_bayesian/notebook_bayesian_spread.ipynb`

### Environment Setup

Each experiment setting comes with its own environment file located in the corresponding folder:

- Online setting: `experiments/spread/spread.yml`
- Offline setting: `experiments/spread_offline/spread_off.yml`
- Bayesian setting: `experiments/spread_bayesian/spread_bay.yml`

To create the environment for a given setting, run:
```bash
conda env create -f experiments/<folder>/<env_name>.yml
conda activate <env_name>
```
For example, to run the online experiments:
```bash
conda env create -f experiments/spread/spread.yml
conda activate spread
```
The offline experiments require installing **Off-MOO-Bench** from the authors’ public repository: https://github.com/lamda-bbo/offline-moo. The datasets should be downloaded into the folder: `experiments/spread_offline/offline_moo/data/`.

## 📃 Citation
If you find `moospread` useful in your research, please consider citing:
```
@inproceedings{
  hotegni2026spread,
  title={{SPREAD}: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion},
  author={Hotegni, Sedjro Salomon and Peitz, Sebastian},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=4731mIqv89}
}
```



