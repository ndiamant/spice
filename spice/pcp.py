# probabilistic conformal prediction
import torch
import numpy as np
from torch import nn
import torch.distributions as D
from sympy import Interval, Union
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import MinMaxScaler

from spice.utils import (
    MLP, BaseLightning, score_to_q_hat, compute_conformal_metrics,
)


class ConditionalGMM(nn.Module):
    """Conditional Gaussian Mixture Model"""
    def __init__(
        self, input_dim: int, hidden_dim: int, n_mixture: int = 10,
    ):
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim, hidden=hidden_dim, n_hidden=0,
        )
        self.get_mean = nn.Linear(hidden_dim, n_mixture)
        self.get_std = nn.Sequential(nn.Linear(hidden_dim, n_mixture), nn.Softplus())
        self.get_mixture_weight_logits = nn.Linear(hidden_dim, n_mixture)

    def forward(self, x: torch.Tensor) -> D.MixtureSameFamily:
        # run modules
        hidden = self.mlp(x)
        mean = self.get_mean(hidden)
        std = self.get_std(hidden) + 1e-4  # prevents NaNs
        weights = self.get_mixture_weight_logits(hidden)
        # build distribution
        mixer = D.Categorical(logits=weights)
        normal = D.Normal(mean, std)
        return D.MixtureSameFamily(mixer, normal)


def union_from_samples(samples: torch.Tensor, q_hat: float) -> Union:
    return Union(
        *(Interval(s - q_hat, s + q_hat) for s in samples)
    )


class PCP(BaseLightning):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        max_iter: int, lr: float = 1e-3, wd: float = 0,
        n_mixture: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cond_gmm = ConditionalGMM(input_dim, hidden_dim, n_mixture)

    def forward(self, x: torch.Tensor) -> D.MixtureSameFamily:
        return self.cond_gmm(x)

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        gmm = self(x)
        log_p = gmm.log_prob(y.squeeze())
        loss = -log_p.nanmean()
        self.epoch_log(f"{prefix}/loss", loss)
        return loss

    @torch.no_grad()
    def get_filtered_samples(
        self, x: torch.Tensor, k: int = 50, beta: float = 0.2,
    ) -> torch.Tensor:
        gmm = self(x)
        samples = gmm.sample((k,))  # K = 50 x batch_size
        # filter
        densities = gmm.log_prob(samples)  # K = 50 x batch_size
        densities_argsort = densities.argsort(dim=0)
        n_filter = int(k * beta)
        keep_idx = densities_argsort[n_filter:]  # k = 40 x batch_size
        filtered_samples = samples[keep_idx, torch.arange(x.shape[0])]
        return filtered_samples.T

    @torch.no_grad()
    def get_q_hat(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        # https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction/blob/54a31cbfe0c87182cbc4351f1d12a59a65452a40/pcp/pcp.py#L28
        n = y_val.shape[0]
        # sample
        filtered_samples = self.get_filtered_samples(x_val.to(self.device)).to(y_val.device)
        # conformal
        score = (filtered_samples - y_val.view(n, 1)).abs().min(dim=1).values
        return score_to_q_hat(score, alpha)

    @torch.no_grad()
    def get_prediction_intervals(
        self, x: torch.Tensor, q_hat: float, parallel_workers: int = 0,
    ) -> list[Union]:
        # sample
        filtered_samples = self.get_filtered_samples(x).cpu()
        desc = "calculating intervals from samples"
        fn = partial(union_from_samples, q_hat=q_hat)
        fn_in = filtered_samples
        if parallel_workers:
            bands = process_map(
                fn, fn_in, max_workers=parallel_workers, desc=desc,
                chunksize=max(1, min(100, fn_in.shape[0] // (2 * parallel_workers)))
            )
        else:
            bands = list(tqdm(map(fn, fn_in), desc=desc, total=len(fn_in)))
        return bands

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, q_hat: float,
        interval_workers: int = 0,
    ) -> dict[str, float]:
        intervals = self.get_prediction_intervals(x_test.to(self.device), q_hat, interval_workers)
        n = y_test.shape[0]
        covered = torch.zeros(n)
        sizes = torch.empty(n)
        for i, (union, yi) in enumerate(tqdm(
            zip(intervals, y_test),
            desc="calculating coverage and size", total=n,
        )):
            sizes[i] = float(union.measure)
            if union.contains(yi.item()):
                covered[i] = 1
        return compute_conformal_metrics(
            x_test, y_test, sizes=sizes, covered=covered,
        )


def plot_pcp(
    model: PCP, x: torch.Tensor, y: float,
    q_hat: float, ax: plt.Axes = None,
    y_min: float = None, y_max: float = None,
    scaler: MinMaxScaler = None,
):
    # preparation
    if ax is None:
        _, ax = plt.subplots(dpi=150)
    x = x.to(model.device)
    # pick color
    samples = model.get_filtered_samples(x)
    intervals = union_from_samples(samples[0], q_hat)
    size = float(intervals.measure)
    covered = intervals.contains(y)
    success_color = (0.173, 0.627, 0.173) if covered else (0.839, 0.153, 0.157)
    plot_y = y
    if scaler is not None:
        plot_y = scaler.transform([[y]])
        size /= scaler.data_range_.item()
    ax.axvline(plot_y, color=success_color, ls="--", zorder=5)
    # plot samples
    with torch.no_grad():
        gmm = model(x)
    sample_likelihood = gmm.log_prob(samples).exp().squeeze().cpu()
    samples = samples.squeeze().cpu()
    if scaler is not None:
        sample_likelihood *= scaler.data_range_
        samples = scaler.transform(samples.unsqueeze(1)).squeeze()
    ax.scatter(samples, sample_likelihood, c="k", label="samples", zorder=0)
    # plot density
    y_min = y_min or samples.min()
    y_max = y_max or samples.max()
    x_eval = torch.linspace(y_min, y_max, 1000, device=model.device).unsqueeze(0)
    density = gmm.log_prob(x_eval).exp().squeeze().cpu()
    x_eval = x_eval.cpu().squeeze()
    x_plot = x_eval
    if scaler is not None:
        density *= scaler.data_range_
        x_plot = scaler.transform(x_eval.unsqueeze(1)).squeeze()
    ax.plot(
        x_plot, density, c="k", zorder=4,
    )
    # plot intervals
    x_covered = torch.zeros_like(x_eval)
    for i, xi in enumerate(x_eval):
        x_covered[i] = float(bool(intervals.contains(xi.item())))
    ymin, ymax = ax.get_ylim()
    y1 = np.repeat(ymin, len(x))
    y2 = np.repeat(ymax, len(x))
    ax.fill_between(
        x_plot, y1, y2,
        color=success_color, alpha=0.1,
        label="$|\\mathcal{{C}}(x)| = {0:.2f}$".format(size),
        where=x_covered,
    )
    ax.set_xlabel("$y$")
    ax.set_ylabel("$\\hatf_{\\theta}(y \\mid x)$")
    ax.legend()
