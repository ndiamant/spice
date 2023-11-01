from itertools import product

from spice.datasets import DATASET_NAMES
from spice.utils import build_full_cmd


VERSION = "09-21-2023_chr_v0"
QUEUE = "short"
CPUS = 8
DURATION = "-W 01:00"
MODEL_NAME = "chr"

wds = [1e-4]
lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
n_bins = [50, 200, 350, 500]
seeds = [1111521, 12341, 5555]

for (
    lr, wd, dataset, seed, n_bin,
) in product(
    lrs, wds, DATASET_NAMES, seeds, n_bins,
):
    print(build_full_cmd(
        cpus=CPUS, queue=QUEUE, duration=DURATION, run_test=False,
        model_name=MODEL_NAME, seed=seed, version=VERSION, dataset_name=dataset,
        extra_python_kwargs={
            "lr": lr, "wd": wd, "n_bins": n_bin,
        },
    ))
