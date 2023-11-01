from itertools import product

from spice.datasets import DATASET_NAMES
from spice.utils import build_full_cmd


VERSION = "09-22-2023_spiceN2_v0"

QUEUE = "short"
CPUS = 4

wds = [1e-4]
lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
n_knots = [11, 21, 31, 51]
learn_bin_widths = [True]
seeds = [1111521, 12341, 5555]
min_f_bars = [1e-3]

for (
    lr, wd, dataset, seed, n_knot, learn_bin_width, min_f_bar,
) in product(
    lrs, wds, DATASET_NAMES, seeds, n_knots, learn_bin_widths, min_f_bars,
):
    print(build_full_cmd(
        cpus=CPUS, queue=QUEUE, duration="00:10", run_test=False,
        model_name="spice_n2", seed=seed, version=VERSION,  dataset_name=dataset,
        extra_python_kwargs={
            "lr": lr, "wd": wd, "min_f_bar": min_f_bar, "n_knots": n_knot,
            "learn_bin_widths": learn_bin_width,
        },
    ))
