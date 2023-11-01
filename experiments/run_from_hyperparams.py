import os
from pathlib import Path
from itertools import product

import pandas as pd

from spice.utils import build_full_cmd


shared_version = "10-03_eval_v0"
seeds = [
    11, 111, 1111, 11111, 111111,
    7, 77, 777, 7777, 77777,
    3, 33, 333, 3333, 33333,
    9, 99, 999, 9999, 99999,
]

dfs = {
    "cqr": "cqr",  # works
    "pcp": "pcp",  # works
    "chr": "chr",  # works
    "conditional_hist": "cond_hist",  # works
    "conditional_hist_HPD": "cond_hist_hpd",  # works
    "spice_n2": "spice_n2",  # works
    "spice_n2_HPD": "spice_n2_hpd",  # works
    "spice_n1": "spice_n1",  # works
    "spice_n1_HPD": "spice_n1_hpd",  # works
}
hp_folder = os.path.join(Path(__file__).parent, "hyperparameters")
dfs = {
    model: pd.read_csv(os.path.join(
        hp_folder, f"{csv}.csv",
    )).set_index("dataset_name")
    for model, csv in dfs.items()
}
dfs["spice_n2"].rename(columns={"min_f_bar_val": "min_f_bar"}, inplace=True)
dfs["spice_n2_HPD"].rename(columns={"min_f_bar_val": "min_f_bar"}, inplace=True)


QUEUE = "short"
CPUS = 8


for model_name, df in dfs.items():
    for seed, (dataset_name, hyperparams) in product(seeds, df.iterrows()):
        version = f"{shared_version}_{model_name}"
        model = model_name.replace("_HPD", "")
        extra_python_kwargs = {}
        for name, val in hyperparams.items():
            if name in {"n_bins", "n_mixture", "n_knots"}: val = int(val)
            extra_python_kwargs[name] = val
        print(build_full_cmd(
            duration="01:00", cpus=CPUS, queue=QUEUE,
            model_name=model, seed=seed, version=version, dataset_name=dataset_name,
            run_test=True, extra_python_kwargs=extra_python_kwargs,
        ))
