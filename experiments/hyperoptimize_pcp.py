from itertools import product

from spice.datasets import DATASET_NAMES
from spice.utils import build_full_cmd


VERSION = "09-21-2023_pcp_v1"
MODEL_NAME = "pcp"
QUEUE = "short"
CPUS = 8
wds = [1e-4]
lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
n_mixtures = [5, 10, 15, 20]
seeds = [1111521, 12341, 5555]

for (
    lr, wd, dataset, n_mixture, seed,
) in product(
    lrs, wds, DATASET_NAMES, n_mixtures, seeds,
):
    print(build_full_cmd(
        cpus=CPUS, queue=QUEUE, duration="00:15", run_test=False,
        model_name=MODEL_NAME, seed=seed, version=VERSION,  dataset_name=dataset,
        extra_python_kwargs={
            "lr": lr, "wd": wd, "n_mixture": n_mixture,
        },
    ))
