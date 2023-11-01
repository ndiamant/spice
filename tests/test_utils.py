import torch
import pytest
import numpy as np

from spice.utils import (
    wsc, wsc_unbiased, unique_quantile, interleave
)
from spice.datasets import RegressionData, DATASET_NAMES


def test_wsc():
    X = np.random.randn(100, 10)
    covered = np.random.choice([0., 1.], 100, p=[0.1, 0.9])
    wsc_star, v_star, a_star, b_star = wsc(X, covered)
    coverage = wsc_unbiased(X, covered)


@pytest.mark.parametrize(
    "dataset_name", DATASET_NAMES,
)
@pytest.mark.parametrize(
    "first_bin_zero", [True, False],
)
@pytest.mark.parametrize(
    "n_bins", [11, 12, 21, 22, 31, 32, 51, 52],
)
def test_unique_quantile(dataset_name: str, first_bin_zero: bool, n_bins: int):
    dset = RegressionData(dataset_name)
    y = dset.train_dset.tensors[1].squeeze()
    q = unique_quantile(y, n_bins=n_bins, first_bin_zero=first_bin_zero)
    assert q.shape == (n_bins,)
    assert len(q.unique()) == len(q)


def test_unique_quantile_too_few_values():
    y = torch.tensor([1.0] * 100 + [0.0])
    n_bins = 5
    unique_quantile(y, n_bins, verbose=True)

    y = torch.tensor([1.0] * 100)
    with pytest.raises(ValueError):
        unique_quantile(y, n_bins, verbose=True)


def test_interleave():
    a = [
        [1, 1, 1],
        [2, 2, 2],
    ]
    b = [
        [3, 3, 3],
        [4, 4, 4],
    ]
    a, b = torch.tensor(a), torch.tensor(b)
    c = interleave(a, b)
    assert (c == torch.tensor([
        [1, 3, 1, 3, 1, 3],
        [2, 4, 2, 4, 2, 4],
    ])).all()
