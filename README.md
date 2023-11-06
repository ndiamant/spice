# Package for _Spline Prediction Intervals via Conformal Estimation_ (SPICE)

[Pre-print link](https://arxiv.org/abs/2311.00774).

## Installation

Installation uses `mamba` and was tested with `CUDA` Version: 12.1 and `python` 3.10.6.

### Mamba install
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

### SPICE package install

```bash
mamba env create -f environment.yml
python -m pip install -e .
wandb login
```

### Before running
First edit `USERNAME` in `spice/utils.py` to be your username.
Some other hardcoded paths might have to be edited in `build_python_cmd`.
You can edit `build_scheduler_cmd` to build commands that work with your job scheduler.


## Workflow

To reproduce the whole paper, the procedure is as follows:
1. Run all the `hyperoptimize` scripts in the `experiments` directory. This runs the hyperparameter grids for all models.
2. Run `experiments/collect_hyperparameters.py`. This collects the results from the hyperameter search.
3. Run `experiments/run_from_hyperparams.py`. This runs 20 replicates of each model's best hyperparameters.
4. Run `experiments/get_test_results.py`. This builds latex tables from all the test set model runs.

You can also just run steps 3. and 4. using the saved hyperparameters in `experiments/hyperparameters.zip`.
