# Piecewise Quantile Regression Spline
# uses piecewise quadratic splines for the density function


import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

from spice.utils import (
    BaseLightning, MLP, never_nan_log, wsc_unbiased, score_to_q_hat,
    softplus_inverse, unique_quantile, compute_conformal_metrics,
)


def lagrange_coeffs(
    x1: torch.Tensor, y1: torch.Tensor,
    x2: torch.Tensor, y2: torch.Tensor,
    x3: torch.Tensor, y3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x1 ** 2 * (y2 - y3) + x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1)) / denom
    c = (x2 ** 2 * (x3 * y1 - x1 * y3) + x2 * (x1 ** 2 * y3 - x3 ** 2 * y1) + x1 * x3 * (x3 - x1) * y2) / denom
    return a, b, c


def f_bar(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, min_val: float = 0.0) -> torch.Tensor:
    return torch.clip(a * x ** 2 + b * x + c, min_val)


def batch_f_bar(
    x: torch.Tensor,
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    min_val: float = 0.0,
) -> torch.Tensor:
    """
    x: inputs to polynomial: bsz x d
    knot_left: bsz x (K - 1)
    knot_right: bsz x (K - 1)
    a: bsz x K - 1
    b: bsz x K - 1
    c: bsz x K - 1
    where K is the number of knots.

    return: bsz x d
    """
    assert x.ndim == 2
    assert x.shape[0] == knot_left.shape[0]
    assert len({knot_left.shape, knot_right.shape, a.shape, b.shape, c.shape}) == 1

    # find which coefficients to use with which inputs
    which_bin_mask = (
        (x.unsqueeze(-1) >= knot_left.unsqueeze(1))
        & (x.unsqueeze(-1) < knot_right.unsqueeze(1))
    )  # bsz x d x (K - 1)
    a = torch.masked_select(a.unsqueeze(1), which_bin_mask).reshape(x.shape)
    b = torch.masked_select(b.unsqueeze(1), which_bin_mask).reshape(x.shape)
    c = torch.masked_select(c.unsqueeze(1), which_bin_mask).reshape(x.shape)

    # evaluate the function
    return f_bar(x, a, b, c, min_val)


def _quad_integral(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a / 3 * x ** 3 + b / 2 * x ** 2 + c * x


def quad_integral(
    x0: torch.Tensor, x1: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> torch.Tensor:
    return _quad_integral(x1, a, b, c) - _quad_integral(x0, a, b, c)


def f_bar_integral(
    x0: torch.Tensor, x1: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    f_bar_min_value: float = 0,
) -> torch.Tensor:
    # calc integral as if just a quadratic
    above_zero_int = quad_integral(x0, x1, a, b, c)
    # subtract the part below zero, which is thresholded
    discriminant = b ** 2 - 4 * a * (c - f_bar_min_value)
    root_denom = torch.clip((2 * a).abs(), 1e-10) * (torch.sign(a) + 1e-10)
    root_1 = (-b - torch.sqrt(discriminant.clip(1e-10))) / root_denom
    root_2 = (-b + torch.sqrt(discriminant.clip(1e-10))) / root_denom
    smaller_root = torch.minimum(root_1, root_2)
    bigger_root = torch.maximum(root_1, root_2)
    # find the intersection of (x0, x1) ^ (r1, r1)
    smaller_root = torch.minimum(torch.maximum(smaller_root, x0), x1)
    bigger_root = torch.maximum(torch.minimum(bigger_root, x1), x0)
    between_roots_int = quad_integral(smaller_root, bigger_root, a, b, c)
    # set the integral to zero if there are no roots or if the parabola's peak is above 0
    parab_max = c - f_bar_min_value - b ** 2 / (4 * a)
    ignore_roots_integral = (discriminant <= 0) | (parab_max > 0) | (root_1.isnan()) | (root_2.isnan())
    between_roots_int = torch.where(ignore_roots_integral, 0, between_roots_int)
    # account for f_bar_min_value
    min_val_integral = (bigger_root - smaller_root) * f_bar_min_value
    min_val_integral = torch.where(ignore_roots_integral, 0, min_val_integral)
    # return the result
    return above_zero_int - between_roots_int + min_val_integral


@torch.no_grad()
def integrate_above_cutoff(
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    integral_above = batch_f_bar_integral(
        knot_left, knot_right, a, b, c - cutoff,
    )
    intervals_above = get_intervals(knot_left, knot_right, a, b, c, cutoff)
    intervals_above_sizes = get_interval_sizes(knot_left, knot_right, *intervals_above).unsqueeze(1)
    return integral_above + intervals_above_sizes * cutoff


@torch.no_grad()
def max_f_bar_val(
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> torch.Tensor:
    # knot heights
    left_height = batch_f_bar(knot_left, knot_left, knot_right, a, b, c)
    right_height = batch_f_bar(knot_right - 1e-5, knot_left, knot_right, a, b, c)
    max_height = torch.maximum(left_height, right_height)
    # vertex heights
    parab_max = c - b ** 2 / (4 * a)
    parab_max_x = -b / (2 * a)
    vertex_between_knots = (
        (parab_max_x >= knot_left) & (parab_max_x < knot_right)
    )
    parab_max = torch.where(vertex_between_knots, parab_max, 0)
    # merge vertex and knot heights
    return torch.maximum(parab_max, max_height).max(dim=1).values


@torch.no_grad()
def find_hpd_cutoff(
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    target_integral: float, verbose: bool = False,
    max_iter: int = 15,
) -> torch.Tensor:
    lower = torch.zeros_like(knot_left[:, 0]).unsqueeze(1)
    upper = max_f_bar_val(knot_left, knot_right, a, b, c).unsqueeze(1)
    for i in range(max_iter):
        mid = (upper + lower) / 2
        score_mid = integrate_above_cutoff(knot_left, knot_right, a, b, c, mid)
        lower = torch.where(
            score_mid > target_integral, mid, lower,
        )
        upper = torch.where(
            score_mid > target_integral, upper, mid,
        )
        if verbose:
            print(f"{i}: mean integral difference = {(score_mid - target_integral).abs().mean():.4f}")
    return mid


@torch.no_grad()
def get_intervals(
    knot_left: torch.Tensor, knot_right: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    k: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k is the cutoff"""
    # find roots
    c = c - k
    discriminant = b ** 2 - 4 * a * c
    root_1 = (-b - torch.sqrt(discriminant)) / (2 * a)
    root_2 = (-b + torch.sqrt(discriminant)) / (2 * a)
    smaller_root = torch.minimum(root_1, root_2)
    bigger_root = torch.maximum(root_1, root_2)
    vertex_y = c - b ** 2 / (4 * a)

    root_deriv = 2 * a * smaller_root + b
    inside_interval_flag = (root_deriv > 0) & (vertex_y > 0)  # redundant but numerical issues can affect the roots
    left_interval = smaller_root
    right_interval = bigger_root

    # roots don't exist case
    left_interval = torch.where((vertex_y > 0) & torch.isnan(smaller_root), knot_left, left_interval)
    right_interval = torch.where((vertex_y > 0) & torch.isnan(smaller_root), knot_right, right_interval)
    inside_interval_flag = torch.where((vertex_y > 0) & torch.isnan(smaller_root), True, inside_interval_flag)

    # bound intervals by x0 and x1
    left_interval = torch.maximum(knot_left, left_interval)
    left_interval = torch.minimum(left_interval, knot_right)
    right_interval = torch.minimum(knot_right, right_interval)
    right_interval = torch.maximum(right_interval, knot_left)
    return left_interval, right_interval, inside_interval_flag


@torch.no_grad()
def get_interval_sizes(
    x0: torch.Tensor, x1: torch.Tensor, left: torch.Tensor, right: torch.Tensor, inside_indicator: torch.Tensor,
) -> torch.Tensor:
    deltas = right - left
    return torch.where(inside_indicator, deltas, (x1 - x0) - deltas).nansum(dim=1)


@torch.no_grad()
def y_in_interval(
    y: torch.Tensor,
    x0: torch.Tensor, x1: torch.Tensor, left: torch.Tensor, right: torch.Tensor, inside_indicator: torch.Tensor,
) -> torch.Tensor:
    mask = (
        ((y >= left) & (y < right) & inside_indicator)  # between left and right case
        | (  # outside left and right case
            (y >= x0) & (y < x1)
            & ((y >= right) | (y < left))
            & ~inside_indicator
        )
    )
    return mask.any(dim=-1)


def batch_f_bar_integral(
    knot_left: torch.Tensor, knot_right: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    f_bar_min_val: float = 0.0,
):
    """
    knot_left: bsz x (K - 1)
    knot_right: bsz x (K - 1)
    a: bsz x K - 1
    b: bsz x K - 1
    c: bsz x K - 1
    where K is the number of knots.

    return: bsz x 1
    """
    assert knot_left.ndim == 2
    assert len({knot_left.shape, knot_right.shape, a.shape, b.shape, c.shape}) == 1
    integrals = f_bar_integral(knot_left, knot_right, a, b, c, f_bar_min_val).clip(1e-3)
    return integrals.sum(dim=1, keepdim=True)


def all_between_01(x: torch.Tensor, name: str = None) -> bool:
    if not torch.all(x >= -1e-6):
        print(name, "min out of bounds", x.min().item())
        return True
    if not torch.all(x <= 1 + 1e-6):
        print(name, "max out of bounds", x.max().item())
        return True
    return False


def smart_bin_init(y_train: torch.Tensor, n_knots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    return:
    x positions: n_knots - 1
    y_positions: n_knots - 1
    """
    quantiles = unique_quantile(y_train.squeeze(), n_knots)
    heights = torch.histogram(y_train.squeeze(), quantiles, density=True).hist
    widths = quantiles[1:] - quantiles[:-1]
    return widths, heights


class ConditionalQuadratic(nn.Module):
    def __init__(
        self, condition_dim: int, n_knots: int,
        learn_bin_widths: bool = False,
        min_f_bar_val: float = 1e-2,  # This is the minimum likelihood the model can output
        bin_width_init: torch.Tensor = None,
        bin_height_init: torch.Tensor = None,
    ):
        """
        bin_width_init: n_knots - 1
        bin_height_init: n_knots - 1
        """
        super().__init__()
        self.n_knots = n_knots
        self.learn_bin_widths = learn_bin_widths
        self.min_f_bar_val = min_f_bar_val
        self.get_widths = nn.Sequential(
            nn.GELU(), nn.Linear(condition_dim, n_knots - 1),
        ) if learn_bin_widths else None
        self.get_heights = nn.Sequential(
            nn.GELU(), nn.Linear(condition_dim, n_knots * 2 - 1),
        )
        self._init_bins(bin_width_init, bin_height_init)

    @torch.no_grad()
    def _init_bins(self, width_init: torch.Tensor, height_init: torch.Tensor):
        # handle height initialization
        if height_init is None:
            height_init = torch.ones(self.n_knots * 2 - 1)
            height_init[:self.n_knots] = softplus_inverse(torch.ones(self.n_knots))
            height_init += torch.randn_like(height_init) * 1e-2
        else:
            smart_height = torch.zeros(2 * self.n_knots - 1)
            smart_height[:self.n_knots - 1] = softplus_inverse(height_init)
            smart_height[1: self.n_knots] = softplus_inverse(height_init)
            smart_height[self.n_knots:] = height_init
            height_init = smart_height
        self.get_heights[-1].bias = nn.Parameter(height_init)
        self.get_heights[-1].weight = nn.Parameter(
            torch.randn_like(self.get_heights[-1].weight) / 10,
        )
        if width_init is None:
            width_init = torch.full((self.n_knots - 1,), 1 / (self.n_knots - 1))
        if self.learn_bin_widths:
            self.get_widths[-1].bias = torch.nn.Parameter(width_init.log())
            self.get_widths[-1].weight = nn.Parameter(
                torch.randn_like(self.get_widths[-1].weight) / 10,
            )
        else:
            cum_width = width_init.cumsum(dim=0)
            self.register_buffer("x1", F.pad(cum_width, (1, 0))[:-1])
            self.register_buffer("x3", cum_width)
            self.register_buffer("x2", (self.x1 + self.x3) / 2)

    def get_x123(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.learn_bin_widths:
            bsz = z.shape[0]
            x1 = self.x1.repeat(bsz, 1)
            x2 = self.x2.repeat(bsz, 1)
            x3 = self.x3.repeat(bsz, 1)
            return x1, x2, x3
        # widths -> x positions
        w = self.get_widths(z)
        # make sure there's a smallest bin width
        min_bin_width = 1 / (self.n_knots * 10)
        w = F.softmax(w, dim=-1)
        w = min_bin_width + (1 - min_bin_width * self.n_knots) * w
        x = w.cumsum(dim=-1)
        x = torch.cat([torch.zeros_like(x[:, :1]), x], dim=-1)
        x[:, -1] = x[:, -1] + min_bin_width
        x = x.clip(0, 1)
        x1 = x[:, :-1]
        x3 = x[:, 1:]
        x2 = (x3 + x1) / 2
        return x1, x2, x3

    def get_lagrange_inputs(self, z: torch.Tensor) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # x1, x2, x3
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # y1, y2, y3
    ]:
        x1, x2, x3 = self.get_x123(z)
        # normalized heights
        y_all = self.get_heights(z)
        y_positive = F.softplus(y_all[:, :self.n_knots])
        y1 = y_positive[:, :-1]
        y3 = y_positive[:, 1:]
        y2 = y_all[:, self.n_knots:]
        return (x1, x2, x3), (y1, y2, y3)

    def get_quadratic_coeffs(self, z: torch.Tensor) -> tuple[
        tuple[torch.Tensor, torch.Tensor], [torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        (x1, x2, x3), (y1, y2, y3) = self.get_lagrange_inputs(z)
        a, b, c = lagrange_coeffs(x1, y1, x2, y2, x3, y3)
        integral = batch_f_bar_integral(x1, x3, a, b, c, self.min_f_bar_val)
        return (x1, x3), (a / integral, b / integral, c / integral)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        (knot_left, knot_right), (a, b, c) = self.get_quadratic_coeffs(z)
        return batch_f_bar(
            y.clip(0, 1 - 1e-5), knot_left, knot_right, a, b, c, self.min_f_bar_val,
        )


class SPICEn2(BaseLightning):
    def __init__(
        self, input_dim: int, hidden_dim: int, n_knots: int,
        learn_bin_widths: bool,
        max_iter: int, lr: float = 1e-3, wd: float = 0,
        smart_bin_init_w: torch.Tensor = None, smart_bin_init_h: torch.Tensor = None,
        min_f_bar_val: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            MLP(input_dim, hidden=hidden_dim, n_hidden=0),
        )
        self.density = ConditionalQuadratic(
            hidden_dim, n_knots, learn_bin_widths=learn_bin_widths,
            min_f_bar_val=min_f_bar_val,
            bin_width_init=smart_bin_init_w, bin_height_init=smart_bin_init_h,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.density(z, y.clip(0, 1 - 1e-3))

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        likelihood = self(x, y)
        self.epoch_log(f"{prefix}/likelihood", likelihood.mean())
        log_likelihood = never_nan_log(likelihood, eps=1e-5)
        self.epoch_log(f"{prefix}/log_likelihood", log_likelihood.mean())
        self.epoch_log(f"{prefix}/log_likelihood_std", log_likelihood.std(dim=0))
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
    def get_intervals(self, x: torch.Tensor, cutoff: float) -> tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        z = self.encoder(x)
        (x0, x1), (a, b, c) = self.density.get_quadratic_coeffs(z)
        return (x0, x1), get_intervals(x0, x1, a, b, c, cutoff)

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        test_likelihood = self(x_test.to(self.device), y_test.to(self.device))
        covered = (test_likelihood > threshold).float()
        (x0, x1), (left, right, inside) = self.get_intervals(x_test.to(self.device), threshold)
        sizes = get_interval_sizes(x0, x1, left, right, inside)
        metrics = compute_conformal_metrics(x_test, y_test, sizes, covered)
        metrics["approx_size"] = self.approx_size(x_test, threshold)
        return metrics

    @torch.no_grad()
    def approx_size(self, x_test: torch.Tensor, threshold: float):
        y_approx_area = torch.linspace(0, 1, 1000, device=self.device).repeat((x_test.shape[0], 1))
        density_grid = self(
            x_test.to(self.device), y_approx_area,
        )
        return (density_grid > threshold).float().mean().item()

    @torch.no_grad()
    def get_hpd_threshold(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        z = self.encoder(x_val)
        (x0, x1), (a, b, c) = self.density.get_quadratic_coeffs(z)
        y_density = self(x_val, y_val)
        score = integrate_above_cutoff(x0, x1, a, b, c, y_density)
        q_hat = score_to_q_hat(score, alpha)
        return q_hat

    @torch.no_grad()
    def get_hpd_intervals(self, x: torch.Tensor, cutoff: float) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        z = self.encoder(x.to(self.device))
        (x0, x1), (a, b, c) = self.density.get_quadratic_coeffs(z)
        hpd_cutoffs = find_hpd_cutoff(x0.to(self.device), x1.to(self.device), a, b, c, cutoff)
        return (x0, x1, hpd_cutoffs), get_intervals(x0, x1, a, b, c, hpd_cutoffs)

    @torch.no_grad()
    def get_hpd_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, threshold: float,
    ) -> dict[str, float]:
        (x0, x1, cutoffs), intervals = self.get_hpd_intervals(x_test, threshold)
        sizes = get_interval_sizes(x0, x1, *intervals)
        covered = y_in_interval(y_test.to(x0.device), x0, x1, *intervals)
        metrics = compute_conformal_metrics(x_test, y_test, sizes, covered)
        metrics["approx_size"] = self.approx_size(x_test, cutoffs)
        metrics = {
            f"hpd_{name}": val for name, val in metrics.items()
        }
        return metrics


def plot_spice_n2(
    model: SPICEn2, x: torch.Tensor, cutoff: float,
    y: float = None, ax: plt.Axes = None,
):
    """
    x: 1 x d
    """
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
        (left, _, right), _ = model.density.get_lagrange_inputs(z)
    merged_knots = torch.concat([left.squeeze(), right.squeeze()[-1:]]).cpu()
    ax.set_xticks(merged_knots)
    ax.set_xticklabels(["$t_{{{0}}}$".format(i) for i in range(len(merged_knots))])

    ax.grid()
    # interval plotting
    fill_idx = (likelihood > cutoff).cpu().squeeze()
    (x0, x1), (left, right, inside) = model.get_intervals(x, cutoff)
    size = get_interval_sizes(x0, x1, left, right, inside).item()
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


def plot_spice_n2_hpd(
    model: SPICEn2, x: torch.Tensor, cutoff: float,
    y: float = None, ax: plt.Axes = None,
):
    """
    x: 1 x d
    """
    # preparation
    if ax is None:
        _, ax = plt.subplots(dpi=150)
    # density plot
    y_eval = torch.linspace(0, 1 - 1e-3, 250, device=model.device).unsqueeze(0)
    with torch.no_grad():
        likelihood = model(x.to(model.device), y_eval).cpu()
    ax.plot(y_eval.cpu().squeeze().detach(), likelihood.squeeze().detach(), label="$w_{\\phi}(y \\mid x)$", color="k")
    # cutoff component
    with torch.no_grad():
        z = model.encoder(x.to(model.device))
        (knot_left, knot_right), (a, b, c) = model.density.get_quadratic_coeffs(z)
        hpd_cutoff = find_hpd_cutoff(knot_left, knot_right, a, b, c, cutoff).item()
        intervals = get_intervals(knot_left, knot_right, a, b, c, hpd_cutoff)
        size = get_interval_sizes(knot_left, knot_right, *intervals).item()
    merged_knots = torch.concat([knot_left.squeeze(), knot_right.squeeze()[-1:]]).cpu()
    ax.set_xticks(merged_knots)
    ax.set_xticklabels(["$t_{{{0}}}$".format(i) for i in range(len(merged_knots))])
    ax.grid()
    # y component
    if y is not None:
        y_covered = y_in_interval(y, knot_left, knot_right, *intervals).item()
        likelihood_color = "green" if y_covered else "red"
    else:
        likelihood_color = "green"
    ax.axhline(cutoff, ls=(0, (1, 5)), c=likelihood_color, label=f"$U(\\hatq) = {hpd_cutoff:.2f}$")
    # interval plotting
    fill_idx = (likelihood > hpd_cutoff).cpu().squeeze()
    ax.fill_between(
        y_eval.detach().cpu().squeeze(),
        torch.full_like(likelihood.squeeze(), cutoff),
        likelihood.squeeze(),
        color=likelihood_color, alpha=0.1,
        label="$|\\mathcal{{C}}(x)| = {0:.2f}$".format(size),
        where=fill_idx,
    )
    ax.legend()
