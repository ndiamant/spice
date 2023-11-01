import torch

from spice.pcp import ConditionalGMM, PCP


def test_conditional_gmm():
    d = 3
    bsz = 2
    x = torch.randn((bsz, d))
    model = ConditionalGMM(d, 5, 7)
    gmm = model(x)
    p1 = gmm.log_prob(torch.ones(bsz))
    p2 = gmm.log_prob(torch.arange(bsz))
    assert p1[1] == p2[1]
    assert p1[0] != p2[0]


def test_pcp():
    d = 3
    bsz = 23
    x = torch.randn((bsz, d))
    y = torch.randn(bsz)
    model = PCP(d, 5, max_iter=20)
    model.get_loss([x, y], "asdf")
    filt = model.get_filtered_samples(x, k=10, beta=0.2)
    assert filt.shape == (bsz, 8)
    q_hat = model.get_q_hat(x, y, 0.2)
    torch.manual_seed(1111)
    pred_intervals_1 = model.get_prediction_intervals(x, q_hat)
    torch.manual_seed(1111)
    pred_intervals_2 = model.get_prediction_intervals(x, q_hat, parallel_workers=2)
    assert pred_intervals_1 == pred_intervals_2
    metrics = model.get_metrics(x, y, q_hat)
