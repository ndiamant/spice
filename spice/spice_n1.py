# Linear Conditional Density
# An example of Piecewise Polynomial Conditional Density (PiP-CoDe)


import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

from spice.utils import (
    BaseLightning, MLP, never_nan_log, wsc_unbiased, softplus_inverse, unique_quantile, score_to_q_hat,
    compute_conformal_metrics,
)


def linear_interp(
    x: torch.Tensor,
    knot_pos: torch.Tensor,
    knot_height: torch.Tensor,
) -> torch.Tensor:
    """
    x: inputs to piecewise linear: bsz x d
    knot_pos: bsz x K
    knot_height: bsz x K
    where K is the number of knots.
    return: bsz x d
    """
    assert x.ndim == 2
    assert x.shape[0] == knot_pos.shape[0]
    assert len({knot_height.shape, knot_pos.shape}) == 1
    # find which coefficients to use with which inputs
    height_left = knot_height[:, :-1]
    height_right = knot_height[:, 1:]
    knot_left = knot_pos[:, :-1]
    knot_right = knot_pos[:, 1:]
    which_bin_mask = (
        (x.unsqueeze(-1) >= knot_left.unsqueeze(1))
        & (x.unsqueeze(-1) < knot_right.unsqueeze(1))
    )  # bsz x d x (K - 1)
    select_knot_left = torch.masked_select(knot_left.unsqueeze(1), which_bin_mask).reshape(x.shape)
    select_knot_right = torch.masked_select(knot_right.unsqueeze(1), which_bin_mask).reshape(x.shape)
    select_height_left = torch.masked_select(height_left.unsqueeze(1), which_bin_mask).reshape(x.shape)
    select_height_right = torch.masked_select(height_right.unsqueeze(1), which_bin_mask).reshape(x.shape)
    # calculate the linear interpolation
    slope = (select_height_right - select_height_left) / (select_knot_right - select_knot_left)
    delta_x = x - select_knot_left
    delta_y = delta_x * slope
    y = delta_y + select_height_left
    return y


def integrate_piecewise_linear_left_right(
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    height_left: torch.Tensor, height_right: torch.Tensor,
) -> torch.Tensor:
    """helper for integrate_piecewise_linear"""
    return (0.5 * (knot_right - knot_left) * (height_right + height_left)).sum(dim=1, keepdim=True)


def integrate_piecewise_linear(
    knot_pos: torch.Tensor, knot_height: torch.Tensor,
) -> torch.Tensor:
    assert knot_pos.shape == knot_height.shape
    assert knot_pos.ndim == 2
    height_left = knot_height[:, :-1]
    height_right = knot_height[:, 1:]
    knot_left = knot_pos[:, :-1]
    knot_right = knot_pos[:, 1:]
    return integrate_piecewise_linear_left_right(
        knot_left, knot_right, height_left, height_right,
    )


@torch.no_grad()
def integrate_above_cutoff(
    knot_pos: torch.Tensor, knot_height: torch.Tensor, cutoff: float,
) -> torch.Tensor:
    """
    integral of piecewise linear density where the density is above `cutoff`
    O(K) runtime
    """
    left, right = get_intervals(knot_pos, knot_height, cutoff)
    left = left.clip(0, 1 - 1e-5)
    right = right.clip(0, 1 - 1e-5)
    # left, right: bsz x K - 1
    height_left = linear_interp(left, knot_pos, knot_height)
    height_right = linear_interp(right, knot_pos, knot_height)
    return integrate_piecewise_linear_left_right(left, right, height_left, height_right)


@torch.no_grad()
def integrate_below_cutoff(
    knot_pos: torch.Tensor, knot_height: torch.Tensor, cutoff: float,
) -> torch.Tensor:
    return 1.0 - integrate_above_cutoff(knot_pos, knot_height, cutoff)


@torch.no_grad()
def find_hpd_cutoff(
    knot_pos: torch.Tensor, knot_height: torch.Tensor, target_integral: float,
    max_iter: int = 15,
) -> torch.Tensor:
    lower = torch.zeros_like(knot_pos[:, 0]).unsqueeze(1)
    upper = knot_height.max(dim=1, keepdim=True).values
    for i in range(max_iter):
        mid = (upper + lower) / 2
        score_mid = integrate_below_cutoff(knot_pos, knot_height, mid)
        lower = torch.where(
            score_mid < target_integral, mid, lower,
        )
        upper = torch.where(
            score_mid < target_integral, upper, mid,
        )
    return mid


@torch.no_grad()
def get_intervals(knot_pos: torch.Tensor, knot_height: torch.Tensor, cutoff: float) -> tuple[
    torch.Tensor, torch.Tensor,
]:
    """
    Find the intervals where the piecewise linear spline is above the cutoff.
    return:
    left boundary, right boundary of intervals
    """
    knot_height = knot_height - cutoff
    height_left = knot_height[..., :-1]
    height_right = knot_height[..., 1:]
    knot_left = knot_pos[..., :-1]
    knot_right = knot_pos[..., 1:]
    slope = (height_right - height_left) / (knot_right - knot_left)
    # non-zero slope case
    intersect_x = knot_left - height_left / slope  # where each line crosses the cutoff
    intersect_x = intersect_x.clip(knot_left, knot_right)
    right_is_knot_right = slope > 0
    right = torch.where(right_is_knot_right, knot_right, intersect_x)
    left = torch.where(~right_is_knot_right, knot_left, intersect_x)

    # slope zero case
    zero_slope = slope == 0
    above_zero = height_left >= 0
    all_above = zero_slope & above_zero
    # this sets left == knot_left and right == knot_right when slope == 0 and left > 0
    left = torch.where(all_above, knot_left, left)
    right = torch.where(all_above, knot_right, right)
    all_below = zero_slope & ~above_zero
    # this sets left == right when slope == 0 and left < 0
    left = torch.where(all_below, knot_left, left)
    right = torch.where(all_below, knot_left, right)
    #
    return left, right


@torch.no_grad()
def get_interval_sizes(
    left: torch.Tensor, right: torch.Tensor,
) -> torch.Tensor:
    return (right - left).sum(dim=1)


@torch.no_grad()
def smart_bin_init(y_train: torch.Tensor, n_knots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    return:
    x positions: n_knots - 1
    y_positions: n_knots - 1
    """
    quantiles = unique_quantile(y_train.squeeze(), n_knots + 1)
    heights = torch.histogram(y_train.squeeze(), bins=quantiles, density=True).hist
    final_heights = heights
    quantiles = unique_quantile(y_train.squeeze(), n_knots)
    widths = quantiles[1:] - quantiles[:-1]
    return widths, final_heights


class ConditionalPiecewiseLinearDensity(nn.Module):
    def __init__(
        self,
        condition_dim: int, n_knots: int,
        learn_bin_widths: bool = False,
        min_likelihood: float = 1e-2,  # This is the minimum likelihood the model can output
        bin_width_init: torch.Tensor = None,
        bin_height_init: torch.Tensor = None,
    ):
        super().__init__()
        self.n_knots = n_knots
        self.learn_bin_widths = learn_bin_widths
        self.min_likelihood = min_likelihood
        self.get_widths = nn.Sequential(
            nn.GELU(), nn.Linear(condition_dim, n_knots - 1),
        ) if learn_bin_widths else None
        self.get_heights = nn.Sequential(
            nn.GELU(), nn.Linear(condition_dim, n_knots),
        )
        self._init_bins(bin_width_init, bin_height_init)

    @torch.no_grad()
    def _init_bins(self, width_init: torch.Tensor, height_init: torch.Tensor):
        # handle height initialization
        if height_init is None:
            height_init = softplus_inverse(torch.ones(self.n_knots))
            height_init += torch.randn_like(height_init) * 1e-2
        else:
            assert (height_init >= 0).all()
            assert height_init.shape == (self.n_knots,)
            height_init = softplus_inverse(height_init)
        self.get_heights[-1].bias = nn.Parameter(height_init)
        self.get_heights[-1].weight = nn.Parameter(
            torch.randn_like(self.get_heights[-1].weight) / 10,
        )
        if width_init is None:
            width_init = torch.full((self.n_knots - 1,), 1 / (self.n_knots - 1))
        if self.learn_bin_widths:
            assert (width_init > 0).all()
            assert width_init.shape == (self.n_knots - 1,)
            width_init = width_init / width_init.sum()  # ensures widths sum to one
            self.get_widths[-1].bias = torch.nn.Parameter(width_init.log())
            self.get_widths[-1].weight = nn.Parameter(
                torch.randn_like(self.get_widths[-1].weight) / 10,
            )
        else:
            cum_width = width_init.cumsum(dim=0)
            self.register_buffer("knot_pos", F.pad(cum_width, (1, 0)))

    def get_knot_pos(self, z: torch.Tensor) -> torch.Tensor:
        if not self.learn_bin_widths:
            return self.knot_pos.repeat(z.shape[0], 1)
        # widths -> x positions
        w = self.get_widths(z)
        # make sure there's a smallest bin width
        min_bin_width = 1 / (self.n_knots * 10)
        w = F.softmax(w, dim=-1)
        w = min_bin_width + (1 - min_bin_width * self.n_knots) * w
        pos = w.cumsum(dim=-1)
        pos = torch.cat([torch.zeros_like(pos[:, :1]), pos], dim=1)
        pos[:, -1] = pos[:, -1] + min_bin_width  # puts final bin at 1.0
        return pos

    def get_knot_pos_height(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unnormed_height = self.get_heights(z)
        knot_height = F.softplus(unnormed_height).clip(self.min_likelihood)
        knot_pos = self.get_knot_pos(z)
        integral = integrate_piecewise_linear(knot_pos, knot_height)
        return knot_pos, knot_height / integral

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        knot_pos, knot_height = self.get_knot_pos_height(z)
        y_clipped = y.clip(
            knot_pos.min(dim=1, keepdim=True).values, knot_pos.max(dim=1, keepdim=True).values - 1e-5,
        )
        return linear_interp(y_clipped, knot_pos, knot_height)


class SPICEn1(BaseLightning):
    def __init__(
        self, input_dim: int, hidden_dim: int, n_knots: int,
        learn_bin_widths: bool,
        max_iter: int, lr: float = 1e-3, wd: float = 0,
        bin_width_init: torch.Tensor = None, bin_height_init: torch.Tensor = None,
        min_likelihood: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            MLP(input_dim, hidden=hidden_dim, n_hidden=0),
        )
        self.density = ConditionalPiecewiseLinearDensity(
            hidden_dim, n_knots, learn_bin_widths=learn_bin_widths,
            min_likelihood=min_likelihood,
            bin_width_init=bin_width_init, bin_height_init=bin_height_init,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.density(z, y)

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        likelihood = self(x, y)
        self.epoch_log(f"{prefix}/likelihood", likelihood.mean())
        log_likelihood = never_nan_log(likelihood, eps=1e-5)
        self.epoch_log(f"{prefix}/log_likelihood", log_likelihood.mean())
        self.epoch_log(f"{prefix}/log_likelihood_std", log_likelihood.std(dim=0).mean())
        self.epoch_log(f"{prefix}/log_likelihood_min", log_likelihood.min())
        self.epoch_log(f"{prefix}/log_likelihood_max", log_likelihood.max())
        loss = -log_likelihood.mean()
        self.epoch_log(f"{prefix}/loss", loss)
        return loss

    @torch.no_grad()
    def get_threshold(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        score = -self(x_val.to(self.device), y_val.to(self.device))
        q_hat = score_to_q_hat(score, alpha)
        return -q_hat

    @torch.no_grad()
    def get_intervals(self, x: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        knot_pos, knot_height = self.density.get_knot_pos_height(z)
        return get_intervals(knot_pos, knot_height, cutoff)

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        test_likelihood = self(x_test.to(self.device), y_test.to(self.device))
        covered = (test_likelihood > threshold)
        left, right = self.get_intervals(x_test.to(self.device), threshold)
        sizes = get_interval_sizes(left, right)
        return compute_conformal_metrics(x_test, y_test, sizes, covered)

    @torch.no_grad()
    def get_hpd_threshold(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        z = self.encoder(x_val)
        knot_pos, knot_height = self.density.get_knot_pos_height(z)
        y_density = self(x_val, y_val)
        score = integrate_below_cutoff(knot_pos, knot_height, y_density)
        return -score_to_q_hat(-score, alpha)

    @torch.no_grad()
    def get_knots_and_hpd_cutoffs(self, x: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x.to(self.device))
        knot_pos, knot_height = self.density.get_knot_pos_height(z)
        hpd_cutoffs = find_hpd_cutoff(knot_pos.to(self.device), knot_height.to(self.device), cutoff)
        return knot_pos, knot_height, hpd_cutoffs

    @torch.no_grad()
    def get_hpd_intervals(self, x: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
        knot_pos, knot_height, hpd_cutoffs = self.get_knots_and_hpd_cutoffs(x, cutoff)
        return get_intervals(knot_pos, knot_height, hpd_cutoffs)

    @torch.no_grad()
    def get_hpd_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        left, right = self.get_hpd_intervals(x_test, threshold)
        sizes = get_interval_sizes(left, right)
        covered = (
            (y_test >= left.cpu())
            & (y_test < right.cpu())
        ).any(dim=1)
        metrics = compute_conformal_metrics(x_test, y_test, sizes, covered)
        metrics = {
            f"hpd_{name}": val for name, val in metrics.items()
        }
        return metrics


def plot_spice_n1(
    model: SPICEn1, x: torch.Tensor, cutoff: float,
    y: float = None, ax: plt.Axes = None,
):
    # preparation
    if ax is None:
        _, ax = plt.subplots(dpi=150)
    y_eval = torch.linspace(0, 1 - 1e-3, 250, device=model.device).unsqueeze(0)
    # interpolation plotting
    x = x.to(model.device)
    with torch.no_grad():
        likelihood = model(x, y_eval).cpu()
    ax.plot(y_eval.cpu().squeeze().detach(), likelihood.squeeze().detach(), color="k")
    if y is not None:
        y_likelihood = model(x, torch.tensor([[y]], device=model.device)).item()
        likelihood_color = "green" if y_likelihood > cutoff else "red"
    else:
        likelihood_color = "green"
    ax.axhline(cutoff, ls="--", c="gray", label="$-\\hat{q}$")
    if y is not None:
        ax.axvline(y, color=likelihood_color, ls="--")
    ax.set_xlabel("$y$")

    with torch.no_grad():
        z = model.encoder(x)
        knot_pos, knot_height = model.density.get_knot_pos_height(z)
    knot_pos = knot_pos.squeeze().cpu()
    ax.set_xticks(knot_pos)
    ax.set_xticklabels(["$t_{{{0}}}$".format(i) for i in range(len(knot_pos))])

    ax.grid()
    # interval plotting
    fill_idx = (likelihood > cutoff).cpu().squeeze()
    left, right = model.get_intervals(x, cutoff)
    size = get_interval_sizes(left, right).item()
    ymin, ymax = ax.get_ylim()
    ax.fill_between(
        y_eval.detach().cpu().squeeze(),
        torch.full_like(likelihood.squeeze(), ymin),
        torch.full_like(likelihood.squeeze(), ymax),
        color=likelihood_color, alpha=0.1,
        label="$|\\mathcal{{C}}(x)| = {0:.2f}$".format(size),
        where=fill_idx,
    )
    ax.set_ylabel("$\\hatf_{\\theta}(y \\mid x)$")
    ax.legend()


def plot_spice_n1_hpd(
    model: SPICEn1, x: torch.Tensor, cutoff: float,
    y: float = None, ax: plt.Axes = None,
):
    # preparation
    if ax is None:
        _, ax = plt.subplots(dpi=150)
    y_eval = torch.linspace(0, 1 - 1e-3, 250, device=model.device).unsqueeze(0)
    # interpolation plotting
    x = x.to(model.device)
    with torch.no_grad():
        likelihood = model(x, y_eval).cpu()
    ax.plot(y_eval.cpu().squeeze().detach(), likelihood.squeeze().detach(), label="$w_{\\phi}(y \\mid x)$", color="k")
    knot_pos, knot_height, hpd_cutoff = model.get_knots_and_hpd_cutoffs(x, cutoff)
    left, right = get_intervals(knot_pos, knot_height, hpd_cutoff)
    if y is not None:
        covered = (
            (y >= left.cpu())
            & (y < right.cpu())
        ).any()
        likelihood_color = "green" if covered else "red"
    else:
        likelihood_color = "green"
    if y is not None:
        ax.axvline(y, label="true $y$", color=likelihood_color, ls="--")
    ax.axhline(hpd_cutoff.item(), ls=(0, (1, 5)), c=likelihood_color, label="$-\\hat{q}$")
    ax.set_xlabel("$y$")
    knot_pos = knot_pos.squeeze().cpu()
    ax.set_xticks(knot_pos)
    ax.set_xticklabels(["$t_{{{0}}}$".format(i) for i in range(len(knot_pos))])
    ax.grid()
    # interval plotting
    fill_idx = (
        (y_eval.T >= left)
        & (y_eval.T < right)
    ).any(dim=1).cpu().squeeze()
    size = get_interval_sizes(left, right).item()
    ax.fill_between(
        y_eval.detach().cpu().squeeze(),
        torch.zeros_like(likelihood.squeeze()),
        likelihood.squeeze(),
        color=likelihood_color, alpha=0.1,
        label="$|\\mathcal{{C}}(x)| = {0:.2f}$".format(size),
        where=fill_idx,
    )
    ax.legend()
