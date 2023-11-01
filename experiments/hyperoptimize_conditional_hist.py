from itertools import product

from spice.datasets import DATASET_NAMES
from spice.utils import build_full_cmd


VERSION = "09-22-2023_cond_hist_v1"
DURATION = "00:05"
MODEL_NAME = "conditional_hist"
QUEUE = "short"
CPUS = 4

wds = [1e-4]
lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
n_bins = [11, 21, 31, 51]
smart_bin_positions = [False]
seeds = [1111521, 12341, 5555]

for (
    lr, wd, dataset, seed, n_bin, smart_bin,
) in product(
    lrs, wds, DATASET_NAMES, seeds, n_bins, smart_bin_positions,
):
    print(build_full_cmd(
        cpus=CPUS, queue=QUEUE, duration=DURATION, run_test=False,
        model_name=MODEL_NAME, seed=seed, version=VERSION, dataset_name=dataset,
        extra_python_kwargs={
            "lr": lr, "wd": wd,
            "smart_bin_positions": smart_bin, "n_bins": n_bin,
        },
    ))
