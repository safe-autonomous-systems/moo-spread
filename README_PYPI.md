<p align="center">
  <img src="https://github.com/safe-autonomous-systems/moo-spread/raw/main/images/logo_well_spread.png" 
       alt="moospread logo" width="300">
</p>
<!--
<p align="center">
<a href="https://pypi.org/project/moospread/"><img src="https://img.shields.io/pypi/v/advermorel.svg" alt="PyPI version"></a>
</p>
-->

# SPREAD: Sampling-based Pareto front Refinement via Efficient Adaptive Diffusion

> SPREAD is a novel sampling-based approach for multi-objective optimization that leverages diffusion models to efficiently refine and generate well-spread Pareto front approximations. It combines the expressiveness of diffusion models with multi-objective optimization principles to achieve both high convergence to the Pareto front and excellent diversity across the objective space. SPREAD demonstrates competitive performance against state-of-the-art methods while providing a flexible framework for different optimization contexts.

## 🚀 Getting Started

### Installation

```python
conda create -n moospread python=3.11
conda activate moospread
pip install moospread
# To install CUDA‐enabled PyTorch, run (or visit: https://pytorch.org/get-started/locally/):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Or, to install the latest code from GitHub:
```python
conda create -n moospread python=3.11
conda activate moospread
git clone https://github.com/safe-autonomous-systems/moo-spread.git
cd moo-spread
pip install -e .
# To install CUDA‐enabled PyTorch, run (or visit: https://pytorch.org/get-started/locally/):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
n_var = 30
problem = ZDT2(n_var=n_var)

# Initialize the SPREAD solver
solver = SPREAD(
    problem,
    data_size=10000,
    timesteps=5000,
    num_epochs=1000,
    train_tol=100,
    num_blocks=3,
    validation_split=0.1,
    mode="online",
    seed=2026,
    verbose=True
)

# Solve the problem
results = solver.solve(
    num_points_sample=200,
    strict_guidance=False,
    rho_scale_gamma=0.9,
    nu_t=10.0,
    eta_init=0.9,
    num_inner_steps=10,
    lr_inner=0.9,
    free_initial_h=True,
    use_sigma_rep=False,
    kernel_sigma_rep=0.01,
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

<!--
### 📚 Next steps

For more advanced examples (offline mode, Bayesian mode, custom problems), see the full [documentation](https://moospread.readthedocs.io/en/latest/).
-->

## Citation
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
