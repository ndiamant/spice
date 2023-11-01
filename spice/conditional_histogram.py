import copy
import math

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from spice.utils import (
    BaseLightning, MLP, unique_quantile,
    score_to_q_hat, compute_conformal_metrics,
)


def select_bins(y: torch.Tensor, n_bins: int) -> torch.Tensor:
    return unique_quantile(y, n_bins, first_bin_zero=False)


def discretize(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(y.clip(max=bins[-1] - 1e-5), boundaries=bins)


class ConditionalHist(BaseLightning):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        max_iter: int, bins: torch.Tensor,
        y_min: float,
        lr: float = 1e-3, wd: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = nn.Sequential(
            MLP(input_dim, hidden=hidden_dim, n_hidden=1, output_dim=bins.shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """log bin probabilities"""
        return torch.log_softmax(self.module(x), dim=-1)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """log likelihood of y | x"""
        bin_log_probs = self(x)
        return -F.nll_loss(bin_log_probs, y.squeeze(), reduction="none")

    def likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(x, y).exp()

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        loss = -self.log_likelihood(x, y).mean()
        self.epoch_log(f"{prefix}/loss", loss)
        return loss

    @torch.no_grad()
    def find_prob_threshold(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        """
        alpha: mis-classification rate
        anything above threshold in likelihood should be in the prediction set
        https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf
        """
        n = len(y_val)
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        cal_scores = 1 - self.likelihood(x_val.to(self.device), y_val.to(self.device))
        q_hat = torch.quantile(cal_scores, q_level, interpolation="higher").item()
        return 1 - q_hat

    @torch.no_grad()
    def get_extended_bins(self):
        extended_bins = torch.empty(self.hparams.bins.shape[0] + 1)
        extended_bins[0] = self.hparams.y_min
        extended_bins[1:] = self.hparams.bins
        return extended_bins

    @torch.no_grad()
    def get_bin_widths(self) -> torch.Tensor:
        extended_bins = self.get_extended_bins()
        return extended_bins[1:] - extended_bins[:-1]

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        test_prob = self(x_test.to(self.device)).exp().to(y_test.device)
        prediction_set = test_prob > threshold
        covered = (
            (
                F.one_hot(y_test.squeeze(), num_classes=self.hparams.bins.shape[0])
                & prediction_set
            ).any(dim=1)
        ).float()
        bin_sizes = self.get_bin_widths()
        sizes = (bin_sizes.unsqueeze(0) * prediction_set).sum(dim=1)
        return compute_conformal_metrics(x_test, y_test.float() / y_test.max().item(), sizes, covered)

    @torch.no_grad()
    def get_hpd_threshold(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        all_probs = self(x_val).exp()
        y_probs = all_probs.gather(index=y_val, dim=1)
        bin_sizes = self.get_bin_widths()
        score = integrate_categorical_below_threshold(all_probs.cpu(), y_probs.cpu(), bin_sizes.cpu())
        return -score_to_q_hat(-score, alpha)

    @torch.no_grad()
    def get_hpd_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        # HPD
        probs = self(x_test.to(self.device)).exp().cpu()
        bin_sizes = self.get_bin_widths()
        hpd_cutoffs = find_hpd_cutoffs(probs, bin_sizes.cpu(), threshold)
        bin_mask = probs >= hpd_cutoffs.unsqueeze(1)
        # size
        sizes = (bin_sizes.unsqueeze(0) * bin_mask).sum(dim=1)
        y_onehot = F.one_hot(y_test.squeeze(), num_classes=self.hparams.bins.shape[0])
        covered = (y_onehot & bin_mask).any(dim=1).float()
        # coverage
        metrics = compute_conformal_metrics(x_test, y_test.float() / y_test.max().item(), sizes, covered)
        metrics = {
            f"hpd_{name}": val for name, val in metrics.items()
        }
        return metrics


@torch.no_grad()
def integrate_categorical_below_threshold(
    probs: torch.Tensor, thresholds: torch.Tensor,
    bin_sizes: torch.Tensor,
) -> torch.Tensor:
    assert thresholds.shape == (probs.shape[0], 1)
    assert bin_sizes.shape == (probs.shape[1],)
    integral_below = probs * (probs <= thresholds) * bin_sizes.unsqueeze(0)
    return integral_below.sum(dim=1)


@torch.no_grad()
def find_hpd_cutoffs(
    probs: torch.Tensor, bin_sizes: torch.Tensor, target_integral: float,
) -> torch.Tensor:
    """
    our goal is to find T s.t.:
        (probs[probs < T] * bin_sizes[probs < T]).sum() > target_integral
    """
    bin_densities = probs * bin_sizes.unsqueeze(0)
    sorted_probs, sort_idx = probs.sort(dim=1)
    sorted_bin_densities = bin_densities.gather(index=sort_idx, dim=1)
    integrated_bin_densities = sorted_bin_densities.cumsum(dim=1)
    first_integral_above_idx = (integrated_bin_densities > target_integral).float().argmax(dim=1, keepdim=True)
    return sorted_probs.gather(index=first_integral_above_idx, dim=1).squeeze()


def plot_conditional_hist(
    model: ConditionalHist, x: torch.Tensor, y: int,
    threshold: float, ax: plt.Axes = None,
):

    if ax is None:
        _, ax = plt.subplots(dpi=150)
    x = x.to(model.device)
    with torch.no_grad():
        probs = model(x).exp().cpu().squeeze().numpy()
    grey = (0.498, 0.498, 0.498)
    blue = (0.1216, 0.467, 0.7059)
    colors = [
        (
            blue if probs[i] > threshold
            else grey
        )
        for i in range(probs.shape[0])
    ]
    success_color = (0.173, 0.627, 0.173) if probs[y] > threshold else (0.839, 0.153, 0.157)
    edge_colr = (0.2, 0.2, 0.2)
    colors[y] = success_color
    extended_bins = model.get_extended_bins().cpu().numpy()
    widths = extended_bins[1:] - extended_bins[:-1]
    ax.bar(extended_bins[:-1], probs, width=widths, edgecolor=edge_colr, align="edge", color=colors)
    threshold_handle = ax.axhline(threshold, ls="--", color=tuple(c * 0.8 for c in success_color), label="Threshold")
    ax.set_ylabel("probability")
    ax.set_xlabel("$y$ bin")

    blue_patch = mpatches.Patch(facecolor=blue, label='Prediction set', edgecolor=edge_colr)
    gray_patch = mpatches.Patch(facecolor=grey, label='Not in prediction set', edgecolor=edge_colr)
    green_patch = mpatches.Patch(facecolor=success_color, label='True $y$', edgecolor=edge_colr)
    ax.legend(handles=[blue_patch, gray_patch, green_patch, threshold_handle])


