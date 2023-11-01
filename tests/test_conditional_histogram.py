import torch

from spice.conditional_histogram import (
    select_bins, discretize, ConditionalHist, integrate_categorical_below_threshold, find_hpd_cutoffs,
)


def test_select_bins():
    y = torch.linspace(0, 1, 100)
    bins = select_bins(y, n_bins=5)
    binned = discretize(y, bins)
    _, counts = torch.unique(binned, return_counts=True)
    # make sure bins equally divide the data
    assert len(set(counts.numpy())) == 1


def test_discretize():
    n_bins = 5
    bins = torch.linspace(0, 1, n_bins)
    y = torch.tensor([
        [0, 0.3, 0.9],
        [0.05, 0.31, 0.91],
    ])
    assert (discretize(y, bins) == torch.tensor([
        [0, 2, 4],
        [1, 2, 4],
    ])).all()


def test_conditional_hist():
    d = 5
    bsz = 2
    n_bins = 7
    m = ConditionalHist(d, 3, max_iter=10, bins=torch.linspace(0, 1, n_bins), y_min=0)
    x = torch.randn((bsz, d))
    pred = m(x)
    assert torch.allclose(pred.exp().sum(dim=-1), torch.ones_like(pred[:, 0]))
    assert pred.shape == (bsz, n_bins)
    y = torch.randint(n_bins, size=(bsz,))
    ll = m.log_likelihood(x, y)
    assert ll.shape == (bsz,)
    m.get_loss([x, y], "asdf")


def test_integrate_categorical_below_threshold():
    probs = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.4, 0.5, 0.1],
    ])
    thresholds = torch.tensor([0.4, 0.1]).unsqueeze(1)
    bin_sizes = torch.tensor([2.0, 2.0, 10.0])
    ints = integrate_categorical_below_threshold(probs, thresholds, bin_sizes)
    assert torch.allclose(
        ints, torch.tensor([
            0.8, 1.0,
        ])
    )


def test_find_hpd_cutoff():
    probs = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.4, 0.5, 0.1],
    ])
    target_integral = 0.4
    bin_sizes = torch.tensor([2.0, 2.0, 10.0])
    cutoffs = find_hpd_cutoffs(probs, bin_sizes, target_integral)
    assert (cutoffs == torch.tensor([0.3, 0.1])).all()
