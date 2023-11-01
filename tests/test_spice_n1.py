import pytest
import torch

from spice.spice_n1 import (
    linear_interp, integrate_piecewise_linear, get_intervals, get_interval_sizes, ConditionalPiecewiseLinearDensity,
    integrate_above_cutoff, find_hpd_cutoff, integrate_below_cutoff,
)


def test_linear_interp():
    x = torch.Tensor([
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
    ])
    knot_pos = torch.tensor([
        [0, 0.5, 1.0],
    ]).repeat(2, 1)
    knot_height = torch.tensor([
        [0, 0.5, 1],
        [1, 0.5, 0.1],
    ])
    y = linear_interp(
        x, knot_pos, knot_height,
    )
    y_desired = torch.tensor([
        [0.1, 0.5, 0.9],
        [0.75, 0.5, 0.3],
    ])
    assert torch.allclose(y, y_desired)


def test_integrate_piecewise_linear():
    torch.manual_seed(11)
    knot_height = torch.rand((1, 11))
    knot_pos = torch.rand(11).sort(dim=0).values.unsqueeze(0)
    knot_pos[0, 0] = 0
    knot_pos[0, -1] = 1
    x = torch.linspace(0, 1 - 1e-5, 5000).unsqueeze(0)
    y = linear_interp(x, knot_pos, knot_height)
    numeric_integral = y.mean(dim=1, keepdim=True)
    integral = integrate_piecewise_linear(knot_pos, knot_height)
    assert abs(numeric_integral.item() - integral.item()) < 1e-3


def test_get_intervals_and_sizes():
    knot_pos = torch.tensor([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    knot_height = torch.tensor([0.0, 1.0, 2.0, 2.0, 0.0, 0.2, 0.2])
    cutoff = 0.5
    left, right = get_intervals(knot_pos, knot_height, cutoff)
    sizes = get_interval_sizes(left.unsqueeze(0), right.unsqueeze(0))
    x = torch.linspace(0, 6 - 1e-5, 5000)
    y = linear_interp(x.unsqueeze(0), knot_pos.unsqueeze(0), knot_height.unsqueeze(0))
    approx_size = (y > cutoff).float().mean().item() * 6
    assert abs(sizes.sum().item() - approx_size) < 1e-3


def test_integrate_above_cutoff():
    knot_pos = torch.tensor([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unsqueeze(0) / 6
    knot_height = torch.tensor([0.0, 1.0, 2.0, 2.0, 0.0, 0.2, 0.2]).unsqueeze(0)
    cutoff = 0.5
    integral = integrate_above_cutoff(knot_pos, knot_height, cutoff).item()
    # numeric integral to compare against
    x = torch.linspace(0, 1 - 1e-5, 5000).unsqueeze(0)
    y = linear_interp(x, knot_pos, knot_height)
    y = y[y > cutoff]
    left, right = get_intervals(knot_pos, knot_height, cutoff)
    size = get_interval_sizes(left, right).item()
    numeric_integral = y.mean().item() * size
    assert abs(numeric_integral - integral) < 1e-3


@pytest.mark.parametrize(
    "learn_bin_widths", [False, True],
)
def test_conditional_piecewise_linear_density(learn_bin_widths: bool):
    hidden = 11
    bin_width_init = torch.tensor([0.5, 0.1, 0.1, 0.3])
    bin_height_init = torch.tensor([0.3, 0.2, 0.1, 0.05, 0.4])
    model = ConditionalPiecewiseLinearDensity(
        hidden, 5, bin_width_init=bin_width_init, bin_height_init=bin_height_init,
        learn_bin_widths=learn_bin_widths,
    )
    z = torch.randn((1, hidden))
    y = (torch.cumsum(bin_width_init, dim=0)).unsqueeze(0)

    pred = model(z, y)
    assert pred.shape == y.shape
    knot_pos, knot_height = model.get_knot_pos_height(z)
    integral = integrate_piecewise_linear(knot_pos, knot_height)
    assert abs(integral.item() - 1) < 1e-5


def test_find_hpd_cutoff():
    knot_pos = torch.tensor([
        [0, 0.5, 1.0],
    ]).repeat(2, 1)
    knot_height = torch.tensor([
        [0, 2, 0],
        [2, 0, 2],
    ])
    target_integral = 0.25
    mid = find_hpd_cutoff(
        knot_pos, knot_height, target_integral=target_integral,
    )
    integral = integrate_below_cutoff(knot_pos, knot_height, mid)
    assert ((integral - target_integral).abs() < 1e-4).all()
    left, right = get_intervals(knot_pos, knot_height, mid)
    desired_left = torch.tensor([
        [0.25, 0.5],
        [0.0, 0.75],
    ])
    desired_right = torch.tensor([
        [0.5, 0.75],
        [0.25, 1.0],
    ])
    assert (left - desired_left).abs().max().item() < 1e-3
    assert (right - desired_right).abs().max().item() < 1e-3
