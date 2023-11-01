from pytorch_lightning import seed_everything
import torch

from spice.chr import CHR


def test_chr():
    seed_everything(1111)
    d = 5
    bsz = 91
    n_bins = 7
    m = CHR(d, 3, max_iter=10, n_bins=n_bins)
    x = torch.randn((bsz, d))
    y = torch.rand((bsz, 1))
    pred = m(x)
    m.get_loss([x, y], "asdf")
    alpha = 0.4
    chr_ = m.calibrate(x, y, alpha)
    metrics = m.get_metrics(x, y, chr_)
    coverage = metrics["coverage"]
    assert abs(coverage - (1 - alpha)) < 0.05
