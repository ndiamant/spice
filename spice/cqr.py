# implementation developed from https://github.com/yromano/cqr/blob/master/cqr/torch_models.py

import torch
from torch import nn

from spice.utils import MLP, BaseLightning, score_to_q_hat, compute_conformal_metrics


class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles: torch.Tensor):
        """ Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.register_buffer("quantiles", quantiles)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        # TODO: could be vectorized
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class CQR(BaseLightning):
    """conformalized quantile regression"""
    def __init__(
        self, input_dim: int, hidden_dim: int,
        low_quantile: float, high_quantile: float,
        max_iter: int, lr: float = 1e-3, wd: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = MLP(
            input_dim=input_dim, hidden=hidden_dim, n_hidden=1, output_dim=2,
        )
        self.quantiles = torch.tensor([low_quantile, high_quantile])
        self.loss_fn = AllQuantileLoss(quantiles=self.quantiles)

    def forward(self, x, sort: bool = False) -> torch.Tensor:
        qs = self.module(x)
        if sort:
            qs = qs.sort(dim=-1).values
        return qs

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.epoch_log(f"{prefix}/loss", loss)
        return loss

    @torch.no_grad()
    def conformity_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred_quantiles = self(x, sort=True)
        lower = pred_quantiles[:, 0] - y.squeeze()
        upper = y.squeeze() - pred_quantiles[:, 1]
        return torch.maximum(lower, upper)

    @torch.no_grad()
    def get_q_hat(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float) -> float:
        conf_score = self.conformity_score(x_val.to(self.device), y_val.to(self.device))
        q_hat = score_to_q_hat(conf_score, alpha)
        return q_hat

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, q_hat: float,
    ) -> dict[str, float]:
        pred_quantiles = self(x_test.to(self.device), sort=True).cpu()
        left_interval = pred_quantiles[:, 0] - q_hat
        right_interval = pred_quantiles[:, 1] + q_hat
        covered = (
            (y_test.squeeze() > left_interval)
            & (y_test.squeeze() < right_interval)
        )
        sizes = (right_interval - left_interval)
        return compute_conformal_metrics(x_test, y_test, sizes=sizes, covered=covered)

    @torch.no_grad()
    def prediction_interval(
        self, x: torch.Tensor, conformity_score: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """alpha is mis-coverage rate"""
        pred_quantiles = self(x, sort=True)
        n_calibrate = conformity_score.shape[0]
        quantile = conformity_score.quantile(
            (1 - alpha) * (1 + 1 / n_calibrate)
        )
        pred_quantiles[:, 0] -= quantile
        pred_quantiles[:, 1] += quantile
        return pred_quantiles
