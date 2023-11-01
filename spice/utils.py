import math
from time import perf_counter
from contextlib import contextmanager
from abc import abstractmethod
from datetime import datetime
from collections import defaultdict
from typing import Iterator, Any

import torch
from torch import nn
from torch import optim
from pytorch_lightning.loggers import Logger
from pytorch_lightning import LightningModule
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# training script stuff
WANDB_PROJECT = "spice"
USERNAME = "YOUR_USERNAME_HERE"
WANDB_LOGDIR = f"/home/{USERNAME}/scratch/spice_logs"
CHECKPOINT_FOLDER = f"/home/{USERNAME}/scratch/spice_checkpoints"


def build_scheduler_cmd(duration: str, cpus: int, queue: str) -> str:
    bsub_log_folder = f"/home/{USERNAME}/scratch/bsub_logs"
    bsub_cmd = [
        f"bsub -o {bsub_log_folder}/output.%J",
        f"-e {bsub_log_folder}/output.%J",
        f"-W {duration}",
        f"-gpu \"num=1:j_exclusive=yes\" -n {cpus} -q {queue}",
    ]
    bsub_cmd = " ".join(bsub_cmd)
    return bsub_cmd


def build_python_cmd(
    model_name: str, seed: int, version: str, dataset_name: str,
    run_test: bool, extra_kwargs: dict[str, Any],
) -> str:
    python_cmd = [
        f"source /home/{USERNAME}/miniforge3/etc/profile.d/conda.sh",
        "&& conda activate spice",
        f"&& python /home/{USERNAME}/repos/spice/experiments/train_eval.py",
        model_name,
        "--seed", seed,
        "--version", version,
        "--epochs", 500,
        "--hidden", 32,
        "--wandb_log_dir", WANDB_LOGDIR,
        "--dataset_name", dataset_name,
        "--alphas", 0.05, 0.1, 0.15,
        "--checkpoint_folder", CHECKPOINT_FOLDER,
        "--run_test", run_test,
    ]
    for arg, val in extra_kwargs.items():
        python_cmd.append(f"--{arg} {val}")
    python_cmd = " ".join(map(str, python_cmd))
    return python_cmd


def build_full_cmd(
    duration: str, cpus: int, queue: str,
    model_name: str, seed: int, version: str, dataset_name: str,
    run_test: bool, extra_python_kwargs: dict[str, Any],
) -> str:
    sched_cmd = build_scheduler_cmd(duration, cpus, queue)
    python_cmd = build_python_cmd(model_name, seed, version, dataset_name, run_test, extra_python_kwargs)
    full_cmd = f'{sched_cmd} "{python_cmd}"'
    return full_cmd


# end training script stuff


def wsc(
    X: np.ndarray, covered: np.ndarray,
    delta: float = 0.1, M: int = 1000, random_state: int = 2023,
):
    """
    Worst Slab Coverage
    covered: array of 1 if covered, 0 if not covered
    """
    rng = np.random.default_rng(random_state)

    def wsc_v(X, cover, delta, v):
        n = len(cover)
        z = np.dot(X, v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        bi_best = min(bi_best, n - 1)
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    for m in tqdm(range(M), desc="computing WSC"):
        wsc_list[m], a_list[m], b_list[m] = wsc_v(X, covered, delta, V[m])

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(
    X: np.ndarray, covered: np.ndarray,
    test_size=0.5, random_state=2020, delta: float = 0.1, M: int = None,
):
    M = M or max(500000 // len(X), 100)

    def wsc_vab(X, cover, v, a, b):
        z = np.dot(X, v)
        idx = np.where((z >= a) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage
    X_train, X_test, covered_train, covered_test = train_test_split(
        X, covered, test_size=test_size, random_state=random_state,
    )
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, covered_train, delta=delta, M=M, random_state=random_state)
    # Estimate coverage
    coverage = wsc_vab(X_test, covered_test, v_star, a_star, b_star)
    return coverage


@contextmanager
def catch_time(name: str) -> float:
    print(f'Beginning {name}...')
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'{name} time: {perf_counter() - start:.3f} seconds')


def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S-%f")


class InMemoryLogger(Logger):
    """somewhat hacky way to use lightning to track metrics in memory for quick experiments"""
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)
        self.hps = None

    def log_metrics(self, metrics: dict[str, float], step: int = None):
        for metric, value in metrics.items():
            self.metrics[metric].append(value)

    @property
    def name(self):
        return "in_memory_logger"

    def log_hyperparams(self, params: dict[str, float], **kwargs):
        self.hps = params

    @property
    def version(self):
        return "0.0"


def never_nan_log(
    x: torch.Tensor, eps: float = 1e-20,
) -> torch.Tensor:
    return torch.clip(x, eps).log()


class BaseLightning(LightningModule):
    def _configure_optimizers(self, parameters: Iterator[torch.nn.Parameter]):
        opt = optim.AdamW(
            parameters, lr=self.hparams.lr, weight_decay=self.hparams.wd,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_iter)
        return [opt], [{"scheduler": scheduler, "interval": "step"}]

    def configure_optimizers(self):
        return self._configure_optimizers(self.parameters())

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return self.get_loss(batch, "train")

    def validation_step(self, batch: list[torch.Tensor], *args) -> torch.Tensor:
        return self.get_loss(batch, "val")

    @abstractmethod
    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        pass

    def epoch_log(
        self,
        name: str,
        value: torch.Tensor,
    ) -> None:
        super().log(name, value, on_epoch=True, on_step=False)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_hidden: int, output_dim: int = None):
        super().__init__()
        output_dim = output_dim or hidden
        self.model = nn.Sequential(
            nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU()),
        )
        for _ in range(n_hidden):
            self.model.append(
                nn.Sequential(nn.Linear(hidden, hidden), nn.GELU()),
            )
        self.model.append(nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def score_to_q_hat(score: torch.Tensor, alpha: float) -> float:
    n = score.shape[0]
    quantile = math.ceil((n + 1) * (1 - alpha)) / n
    q_hat = score.quantile(quantile).item()
    return q_hat


def unique_quantile(
    x: torch.Tensor, n_bins: int, first_bin_zero: bool = True,
    max_try_n_bins: int = None, verbose: bool = False,
) -> torch.Tensor:
    """binary search to find the right number of bins to yield n_bins unique quantiles"""
    if len(x.unique()) == 1:
        raise ValueError("Must have more than one value to find unique quantiles.")

    def _print(x: Any):
        if not verbose:
            return
        print(x)

    min_n_bins = n_bins
    max_try_n_bins = max_try_n_bins or 5 * n_bins
    og_max_try = max_try_n_bins
    unique_quantiles = None
    while min_n_bins <= max_try_n_bins:
        try_n_bins = (min_n_bins + max_try_n_bins) // 2
        first_bin = (0 if first_bin_zero else 1) / try_n_bins
        quantiles = torch.linspace(first_bin, 1, try_n_bins)
        unique_quantiles = torch.unique(x.quantile(quantiles))
        n_unique = unique_quantiles.shape[0]
        _print(f"tried {try_n_bins=} and got {len(unique_quantiles)=} / {n_bins}")
        if n_unique == n_bins:
            _print("found correct number of bins")
            return unique_quantiles
        if n_unique > n_bins:
            max_try_n_bins = try_n_bins - 1
        else:
            min_n_bins = try_n_bins + 1
    if min_n_bins >= og_max_try:
        _print(f"Trying again with 2x max try bins")
        return unique_quantile(
            x, n_bins, first_bin_zero, max_try_n_bins * 2, verbose=verbose,
        )
    _print(f"Algorithm failed, returning closest guess.")
    # likely results in unused bins
    if n_unique < n_bins:
        start, stop = unique_quantiles[-2:]
        lengthened = torch.cat([
            unique_quantiles[:-2],
            torch.linspace(start, stop, n_bins - n_unique + 2)
        ])
        return lengthened
    else:
        deltas = unique_quantiles[1:] - unique_quantiles[:-1]
        min_delta_idx = deltas.argsort()
        idx_to_keep = [
            i for i in list(range(n_unique))
            if i not in min_delta_idx[:n_unique - n_bins]
        ]
        shortened = unique_quantiles[idx_to_keep]
        return shortened


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """https://pytorch.org/rl/_modules/tensordict/nn/utils.html#inv_softplus"""
    return torch.where(
        x > 20, x,
        x.expm1().clamp_min(1e-6).log(),
    )


def interleave(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a = [
        [1, 1, 1],
        [2, 2, 2],
    ]
    b = [
        [3, 3, 3],
        [4, 4, 4],
    ]
    -> [
        [1, 3, 1, 3, 1, 3],
        [2, 4, 2, 4, 2, 4],
    ]
    """
    bsz, d = a.shape
    stack = torch.stack([a, b], dim=2)
    return stack.view(bsz, 2 * d)


def stratified_coverage(
    covered: torch.Tensor, stratifier: torch.Tensor,
) -> float:
    covered = covered.float()
    return min(
        covered[stratifier == i].mean().item()
        for i in stratifier.unique()
    )


def compute_conformal_metrics(
    x_test: torch.Tensor, y_test: torch.Tensor, sizes: torch.Tensor, covered: torch.Tensor,
) -> dict[str, float]:
    x_test = x_test.cpu()
    y_test = y_test.cpu().squeeze()
    sizes = sizes.cpu().squeeze()
    covered = covered.cpu().squeeze()
    metrics = dict()
    metrics["coverage"] = covered.float().mean().item()
    metrics["size"] = sizes.mean().item()
    metrics["wsc_coverage"] = wsc_unbiased(x_test.cpu().numpy(), covered.cpu().numpy())
    # y stratified coverage
    y_quantiles = unique_quantile(y_test, n_bins=5, first_bin_zero=False)
    discrete_y = torch.bucketize(y_test, y_quantiles)
    metrics["y_stratified_coverage"] = stratified_coverage(covered, discrete_y)
    # size stratified coverage
    try:
        size_quantiles = unique_quantile(sizes / sizes.max(), n_bins=5, first_bin_zero=False)
        discrete_size = torch.bucketize(sizes, size_quantiles)
        metrics["size_stratified_coverage"] = stratified_coverage(covered, discrete_size)
    except ValueError:
        pass  # no unique sizes case
    return metrics


def rename_metrics(metrics: dict[str, float], prefix: str, alpha: float) -> dict[str, float]:
    return {
        f"{prefix}/{name}_at_{alpha}": val
        for name, val in metrics.items()
    }
