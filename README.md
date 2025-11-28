# SPREAD: Sampling-based Pareto Front Refinement via Efficient Adaptive Diffusion

### 🔬 Experiments

All experiment code is contained in the `/experiments` directory:

* **Online setting:** `/experiments/spread/`
* **Offline setting:** `/experiments/spread_offline/`
* **Bayesian setting:** `/experiments/spread_bayesian/`

The following Jupyter notebooks reproduce the plots shown in our paper:

* `/experiments/spread/notebook_online_spread.ipynb`
* `/experiments/spread_bayesian/notebook_bayesian_spread.ipynb`

### ⚙️ Environment Setup

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



