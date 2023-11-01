import numpy as np
import pytest

from spice.datasets import get_dataset, DATASET_NAMES, RegressionData


@pytest.mark.parametrize(
    "name", DATASET_NAMES,
)
def test_get_dataset(name: str):
    X, y = get_dataset(name)
    assert X.shape[0] == y.shape[0]
    assert ~np.isnan(y).any()
    assert ~np.isnan(X).any()
    assert y.ndim == 1


@pytest.mark.parametrize(
    "name", DATASET_NAMES,
)
@pytest.mark.parametrize(
    "y_scaling", ["min_max", "std"],
)
def test_regression_data(name: str, y_scaling: str):
    data = RegressionData(name, y_scaling, batch_size=8)
    loader = data.train_dataloader()
    x, y = next(loader.__iter__())
    assert y.shape == (8, 1)


def test_regression_data_shuffle():
    data1 = RegressionData("synthetic_bimodal", train_seed=0)
    data2 = RegressionData("synthetic_bimodal", train_seed=1)
    # check consistent test dataset
    test1_idx = data1.test_idx
    test2_idx = data2.test_idx
    assert (test1_idx == test2_idx).all()
    # check train dataset changes with random seed
    train1_x, train1_y = data1.train_dset.tensors
    train2_x, train2_y = data2.train_dset.tensors
    assert not (train1_x == train2_x).all()
    assert not (train1_y == train2_y).all()

