import torch

from spice.cqr import (
    AllQuantileLoss, CQR,
)


def test_all_quantile_loss():
    quantiles = torch.linspace(0, 1, 5)
    loss_fn = AllQuantileLoss(quantiles)
    y = torch.randn((2,))
    y_pred = torch.randn((2, 5))
    loss_1 = loss_fn(y_pred, y)


def test_cqr():
    bsz = 23
    d = 3
    x = torch.randn((bsz, d))
    y = torch.rand(bsz)
    model = CQR(
        input_dim=d, hidden_dim=2, low_quantile=0.05, high_quantile=0.95, max_iter=301,
    )
    model(x)
    model.get_loss([x, y], "asdf")
    y_pred = model(x)

    conf_score = model.conformity_score(x, y)
    assert conf_score.shape == y.shape
    x2 = torch.randn((11, d))
    y2 = torch.randn((11,))
    pred_interval_1 = model.prediction_interval(
        x2, conf_score, 0.3,
    )
    pred_interval_2 = model.prediction_interval(
        x2, conf_score, 0.4,
    )
    # check that larger interval comes from smaller miscoverage rate
    assert (pred_interval_1[:, 0] < pred_interval_2[:, 0]).all()
    assert (pred_interval_1[:, 1] > pred_interval_2[:, 1]).all()
    q = model.get_q_hat(x, y, 0.1)
    metrics = model.get_metrics(x, y, q)
