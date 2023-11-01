import pytest
import torch

from spice.spice_n2 import (
    batch_f_bar, batch_f_bar_integral, f_bar, f_bar_integral, ConditionalQuadratic, lagrange_coeffs,
    integrate_above_cutoff, max_f_bar_val, find_hpd_cutoff,
)


def test_batch_f_bar():
    bsz = 3
    K = 5
    x = torch.linspace(0, 1 - 1e-3, 7).repeat(bsz, 1)
    a = torch.randn((bsz, K - 1))
    b = torch.randn((bsz, K - 1))
    c = torch.randn((bsz, K - 1))
    knots = torch.linspace(0, 1, K).repeat(bsz, 1)
    knot_left = knots[:, :-1]
    knot_right = knots[:, 1:]
    y = batch_f_bar(x, knot_left, knot_right, a, b, c)
    assert y.shape == x.shape


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_batch_f_bar_integral(seed: int):
    torch.manual_seed(seed)
    bsz = 3
    K = 5
    z_dim = 7
    z = torch.randn((bsz, z_dim))
    quad = ConditionalQuadratic(condition_dim=z_dim, n_knots=K)
    quad.get_heights[-1].bias = torch.nn.Parameter(torch.randn_like(quad.get_heights[-1].bias))
    (knot_left, knot_right), (a, b, c) = quad.get_quadratic_coeffs(z)
    min_f_bar_val = 0.25
    with torch.no_grad():
        integral = batch_f_bar_integral(knot_left, knot_right, a, b, c, min_f_bar_val)
    assert integral.shape == (bsz, 1)
    # make sure integral is correct
    x = torch.linspace(0, 1 - 1e-3, 5000).repeat(bsz, 1)
    with torch.no_grad():
        numeric_integral = batch_f_bar(x, knot_left, knot_right, a, b, c, min_f_bar_val).mean(dim=1)
    assert (integral.squeeze() - numeric_integral).abs().max().item() < 5e-3


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_conditional_quadratic(seed: int):
    torch.manual_seed(seed)
    bsz = 3
    K = 5
    z_dim = 7
    z = torch.randn((bsz, z_dim))
    quad = ConditionalQuadratic(condition_dim=z_dim, n_knots=K)
    quad.get_heights[-1].bias = torch.nn.Parameter(torch.randn_like(quad.get_heights[-1].bias))
    x = torch.linspace(0, 1 - 1e-3, 10000).repeat(bsz, 1)
    density = quad(z, x)
    assert (density >= 0).all()
    assert ((density.mean(dim=1) - 1).abs() < 2e-2).all()


def get_knot_params() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    knot_left = torch.tensor([
        [0.0, 0.5, 0.8],
    ])
    knot_right = torch.tensor([
        [0.5, 0.8, 1.0],
    ])
    knot_mid = (knot_right + knot_left) / 2
    height_left = torch.tensor([
        [1.0, 1.0, 1.0],
    ])
    height_right = torch.tensor([
        [1.0, 1.0, 1.0],
    ])
    height_mid = torch.tensor([
        [-1.0, 1.0, 2.0],
    ])
    a, b, c = lagrange_coeffs(
        knot_left, knot_mid, knot_right,
        height_left, height_mid, height_right,
    )
    return knot_left, knot_right, a, b, c


def test_integrate_above_cutoff():
    knot_left, knot_right, a, b, c = get_knot_params()
    cutoff = 0.5
    integral = integrate_above_cutoff(
        knot_left, knot_right, a, b, c, cutoff,
    ).item()
    # numeric integral
    grid = torch.linspace(0, 1 - 1e-5, 5000).unsqueeze(0)
    y = batch_f_bar(grid, knot_left, knot_right, a, b, c)
    y_above = y[y > cutoff]
    numeric_integral = (y_above * (grid[0, 1] - grid[0, 0])).sum().item()
    assert abs(numeric_integral - integral) < 1e-3


def test_max_f_bar_val():
    knot_left, knot_right, a, b, c = get_knot_params()
    y_max = max_f_bar_val(knot_left, knot_right, a, b, c).item()
    # approx
    grid = torch.linspace(0, 1 - 1e-5, 5000).unsqueeze(0)
    y_max_approx = batch_f_bar(grid, knot_left, knot_right, a, b, c).max().item()
    assert abs(y_max - y_max_approx) < 1e-4


def test_find_hpd_cutoff():
    knot_left, knot_right, a, b, c = get_knot_params()
    target_integral = 0.25
    mid = find_hpd_cutoff(
        knot_left, knot_right, a, b, c, target_integral=target_integral, verbose=True,
    )
    integral = integrate_above_cutoff(knot_left, knot_right, a, b, c, mid)
    assert abs(integral.item() - 0.25) < 1e-4
