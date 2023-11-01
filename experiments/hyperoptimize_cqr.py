from itertools import product

from spice.datasets import DATASET_NAMES
from spice.utils import build_full_cmd


VERSION = "09-21-2023_cqr_v0"
MODEL_NAME = "cqr"
QUEUE = "short"
CPUS = 4
DURATION = "00:10"

wds = [1e-4]
lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
qr_intervals = [0.3, 0.5, 0.7, 0.9]
seeds = [1111521, 12341, 5555]

for (
    lr, wd, dataset, qr_interval, seed,
) in product(
    lrs, wds, DATASET_NAMES, qr_intervals, seeds,
):
    print(build_full_cmd(
        cpus=CPUS, queue=QUEUE, duration=DURATION, run_test=False,
        model_name=MODEL_NAME, seed=seed, version=VERSION, dataset_name=dataset,
        extra_python_kwargs={
            "lr": lr, "wd": wd,
            "qr_interval": qr_interval,
        },
    ))
