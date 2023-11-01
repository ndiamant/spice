import copy

from scipy.stats.mstats import mquantiles
from tqdm.autonotebook import tqdm
import numpy as np
from scipy import interpolate
import torch
from torch import nn
import matplotlib.pyplot as plt

from spice.utils import (
    BaseLightning, compute_conformal_metrics, MLP,
)


# adapted from: https://github.com/msesia/chr/blob/master/chr/black_boxes.py
def _estim_dist(quantiles, percentiles, y_min, y_max, smooth_tails, tau):
    """ Estimate CDF from list of quantiles, with smoothing """

    noise = np.random.uniform(low=0.0, high=1e-5, size=((len(quantiles),)))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    # Smooth tails
    def interp1d(x, y, a, b):
        return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = tau
        tau_hi = 1-tau
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = np.where(percentiles < tau_lo)[0]
        idx_hi = np.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = np.linspace(quantiles[0], q_lo, num=len(idx_lo))
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = np.linspace(q_hi, quantiles[-1], num=len(idx_hi))

        cdf = interp1d(quantiles_smooth, percentiles, 0.0, 1.0)
        inv_cdf = interp1d(percentiles, quantiles_smooth, y_min, y_max)

    # Standardize
    breaks = np.linspace(y_min, y_max, num=1000, endpoint=True)
    cdf_hat = cdf(breaks)
    f_hat = np.diff(cdf_hat)
    f_hat = (f_hat+1e-6) / (np.sum(f_hat+1e-6))
    cdf_hat = np.concatenate([[0],np.cumsum(f_hat)])
    cdf = interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf


class Histogram:
    def __init__(self, percentiles, breaks):
        self.percentiles = percentiles
        self.breaks = breaks

    def compute_histogram(self, quantiles, ymin, ymax, alpha, smooth_tails=True):
        """
        Compute pi_hat[j]: the mass between break[j-1] and break[j]
        """
        n = quantiles.shape[0]
        B = len(self.breaks)-1

        pi_hat = np.zeros((n,B+1))
        percentiles = np.concatenate(([0],self.percentiles,[1]))
        quantiles = np.pad(quantiles, ((0,0),(1, 1)), 'constant', constant_values=(ymin,ymax))

        def interp1d(x, y, a, b):
            return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

        for i in range(n):
            cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=ymin, y_max=ymax,
                                       smooth_tails=smooth_tails, tau=0.01)
            cdf_hat = cdf(self.breaks)
            pi_hat[i] = np.concatenate([[0], np.diff(cdf_hat)])
            pi_hat[i] = (pi_hat[i]+1e-6) / (np.sum(pi_hat[i]+1e-6))

        return pi_hat


def smallestSubWithSum(arr, x, include=None):
    """
    Credit: https://www.geeksforgeeks.org/minimum-length-subarray-sum-greater-given-value/
    """
    n = len(arr)

    # Initialize weights if not provided
    if include is None:
        end_init = 0
        start_max = n
    else:
        end_init = include[1]
        start_max = include[0]

    # Initialize optimal solution
    start_best = 0
    end_best = n
    min_len = n + 1

    # Initialize starting index
    start = 0

    # Initialize current sum
    curr_sum = np.sum(arr[start:end_init])

    for end in range(end_init, n):
        curr_sum += arr[end]
        while (curr_sum >= x) and (start <= end) and (start <= start_max):
            if (end - start + 1 < min_len):
                min_len = end - start + 1
                start_best = start
                end_best = end

            curr_sum -= arr[start]
            start += 1

    if end_best == n:
        print("Error in smallestSubWithSum(): no solution! This may be a bug.")
        quit()

    return start_best, end_best


class HistogramAccumulator:
    def __init__(self, pi, breaks, alpha, delta_alpha=0.001):
        self.n, self.K = pi.shape
        self.breaks = breaks
        self.pi = pi
        self.alpha = alpha

        # Define grid of alpha values
        self.alpha_grid = np.round(np.arange(delta_alpha, 1.0, delta_alpha), 4)

        # Make sure the target value is included
        self.alpha_grid = np.unique(np.sort(np.concatenate((self.alpha_grid,[alpha]))))

        # This is only used to predict sets rather than intervals
        self.order = np.argsort(-pi, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.pi_sort = -np.sort(-pi, axis=1)
        self.Z = np.round(self.pi_sort.cumsum(axis=1),9)

    def compute_interval_sequence(self, epsilon=None):
        alpha_grid = self.alpha_grid
        n_grid = len(alpha_grid)
        k_star = np.where(alpha_grid==self.alpha)[0][0]
        S_grid = -np.ones((n_grid,self.n,2)).astype(int)
        S_grid_random = -np.ones((n_grid,self.n,2)).astype(int)

        # First compute optimal set for target alpha
        S, S_random = self.predict_intervals_single(alpha_grid[k_star], epsilon=epsilon)
        S_grid[k_star] = S
        S_grid_random[k_star] = S_random

        # Compute smaller sets
        for k in tqdm(range(k_star+1, n_grid), desc="Computing smaller sets"):
            a = S_grid[k-1,:,0]
            b = S_grid[k-1,:,1]
            S, S_random = self.predict_intervals_single(alpha_grid[k], epsilon=epsilon, a=a, b=b)
            S_grid[k] = S
            S_grid_random[k] = S_random

        # Compute larger sets
        for k in tqdm(range(0,k_star)[::-1], desc="Computing larger sets"):
            alpha = alpha_grid[k]
            S, S_random = self.predict_intervals_single(alpha, epsilon=epsilon, include=S_grid_random[k+1])
            S_grid[k] = S
            S_grid_random[k] = S_random

        return S_grid_random

    def predict_intervals_single(self, alpha, a=None, b=None, include=None, epsilon=None):
        # I think this function computes S_t
        if a is None:
            a = np.zeros(self.n).astype(int)
        if b is None:
            b = (np.ones(self.n) * len(self.pi[0])).astype(int)

        if include is None:
            include = [None] * self.n

        start = -np.ones(self.n).astype(int)
        end = -np.ones(self.n).astype(int)
        for i in range(self.n):
            start_offset = a[i]
            end_offset = b[i]+1
            if np.sum(self.pi[i][start_offset:end_offset]) < 1.0-alpha:
                print("Error: incorrect probability normalization. This may be a bug.")
                quit()
            start[i], end[i] = smallestSubWithSum(self.pi[i][start_offset:end_offset], 1.0-alpha, include=include[i])
            start[i] += start_offset
            end[i] += start_offset
        S = np.concatenate(
            (
                start.reshape((len(start), 1)),
                end.reshape((len(start), 1))
            ),
            axis=1,
        )
        S_random = copy.deepcopy(S)

        # Randomly remove one bin (either first or last) to seek exact coverage
        if (epsilon is not None):
            for i in range(self.n):
                if((S[i,-1]-S[i,0])<=1):
                    continue
                tot_weight = np.sum(self.pi[i][S[i,0]:(S[i,-1]+1)])
                excess_weight = tot_weight-(1.0-alpha)
                weight_left = self.pi[i][S[i,0]]
                weight_right = self.pi[i][S[i,-1]]
                # Remove the endpoint with the least weight (more likely to be removed)
                if weight_left < weight_right:
                    pi_remove = excess_weight / (weight_left + 1e-5)
                    if epsilon[i] <= pi_remove:
                        S_random[i,0] += 1
                else:
                    pi_remove = excess_weight / (weight_right + 1e-5)
                    if epsilon[i] <= pi_remove:
                        S_random[i,-1] -= 1

        return S, S_random

    def predict_intervals(self, alpha, epsilon=None):
        # Pre-compute list of predictive intervals
        bands = np.zeros((self.n,2))
        S_grid = self.compute_interval_sequence(epsilon=epsilon)
        j_star_idx = np.where(self.alpha_grid <= alpha)[0]
        if len(j_star_idx)==0:
            S = np.tile(np.arange(self.K), (self.n,1))
        else:
            j_star = np.max(j_star_idx)
            S = S_grid[j_star]

        bands[:,0] = self.breaks[np.clip(S[:, 0] - 1, a_min=0, a_max=None)]
        bands[:,1] = self.breaks[S[:,-1]]

        return S, bands

    def calibrate_intervals(self, Y, epsilon=None, verbose=True):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        alpha_max = np.zeros(n2,)
        S_grid = self.compute_interval_sequence(epsilon=epsilon)
        if verbose:
            print("Computing conformity scores.")
        for j in tqdm(range(len(self.alpha_grid)), disable=(not verbose)):
            # iterate through the alpha grid, which is just arange(0, 1, delta_alpha)
            alpha = self.alpha_grid[j]
            S = S_grid[j]
            band_left = self.breaks[np.clip(S[:, 0] - 1, a_min=0, a_max=None)]
            band_right = self.breaks[S[:,-1]]
            y_inside_mask = (Y>=band_left)*(Y<=band_right)
            idx_inside = np.where(y_inside_mask)[0]
            if len(idx_inside)>0:
                alpha_max[idx_inside] = alpha

        return 1.0-alpha_max

    def predict_sets(self, alpha, epsilon=None):
        L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.pi_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                L[i] = L[i] - 1
        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        S = [ np.sort(s) for s in S]
        return(S)

    def calibrate_sets(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        pi_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        pi = np.array([ self.pi_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - pi_cum
        if epsilon is not None:
            alpha_max += np.multiply(pi, epsilon)
        else:
            alpha_max += pi
            alpha_max = np.minimum(alpha_max, 1)
        return alpha_max


class CHRCalibrate:
    """
    Used to calibrate a conditional quantile based model using the CHR method.
    """

    def __init__(self, grid_quantiles: np.ndarray, ymin=0, ymax=1, y_steps=1000, delta_alpha=0.001, intervals=True, randomize=False):

        # Define discrete grid of y values for histogram estimator
        self.grid_histogram = np.linspace(ymin, ymax, num=y_steps, endpoint=True)
        self.ymin = ymin
        self.ymax = ymax

        # Should we predict intervals or sets?
        self.intervals = intervals

        # Store desired nominal level
        self.alpha = None
        self.delta_alpha = delta_alpha

        self.randomize = randomize
        self.hist = Histogram(percentiles=grid_quantiles, breaks=self.grid_histogram)

    def calibrate(self, q_calib, Y, alpha, return_scores=False):
        # Store desired nominal level
        self.alpha = alpha

        # Estimate conditional histogram for calibration points
        d_calib = self.hist.compute_histogram(q_calib, self.ymin, self.ymax, alpha)

        # Initialize histogram accumulator (grey-box)
        accumulator = HistogramAccumulator(d_calib, self.grid_histogram, self.alpha, delta_alpha=self.delta_alpha)

        # Generate noise for randomization
        n2 = q_calib.shape[0]
        if self.randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
        else:
            epsilon = None

        # Compute conformity scores
        if self.intervals:
            scores = accumulator.calibrate_intervals(Y.astype(np.float32), epsilon=epsilon)
        else:
            # TODO: implement this
            assert (1 == 2)

        # Compute upper quantile of scores
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        self.calibrated_alpha = np.round(1.0 - mquantiles(scores, prob=level_adjusted)[0], 4)

        # Print message
        print("Calibrated alpha (nominal level: {}): {:.3f}.".format(alpha, self.calibrated_alpha))

        if return_scores:
            return self.calibrated_alpha, scores
        return self.calibrated_alpha

    def predict(self, q_new: np.ndarray, alpha=None):
        assert (self.alpha is not None)

        # Estimate conditional histogram for new data points
        d_new = self.hist.compute_histogram(q_new, self.ymin, self.ymax, self.alpha)

        # Initialize histogram accumulator (grey-box)
        accumulator = HistogramAccumulator(d_new, self.grid_histogram, self.alpha, delta_alpha=self.delta_alpha)

        # Generate noise for randomization
        n = q_new.shape[0]
        if self.randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        else:
            epsilon = None

        # Compute prediction bands
        if alpha is None:
            alpha = self.calibrated_alpha

        _, bands = accumulator.predict_intervals(alpha, epsilon=epsilon)

        return bands


class AllQuantileLoss(nn.Module):
    """ Pinball loss function """
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
        errors = target.unsqueeze(1) - preds
        Q = self.quantiles.unsqueeze(0)
        loss = torch.max((Q-1.0)*errors, Q*errors).mean()
        return loss


class CHR(BaseLightning):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        max_iter: int, n_bins: int,
        lr: float = 1e-3, wd: float = 0,
        y_min: float = 0, y_max: float = 1,
        hist_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = MLP(input_dim, hidden=hidden_dim, n_hidden=1, output_dim=n_bins)
        self.register_buffer("quantiles", torch.linspace(0.01, 0.99, n_bins))
        self.loss_fn = AllQuantileLoss(quantiles=self.quantiles)

    def forward(self, x: torch.Tensor, sort: bool = True) -> torch.Tensor:
        y = self.mlp(x)
        if sort:
            return y.sort(dim=-1).values
        return y

    def get_loss(self, batch: list[torch.Tensor], prefix: str) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.epoch_log(f"{prefix}/loss", loss)
        return loss

    @torch.no_grad()
    def calibrate(self, x_val: torch.Tensor, y_val: torch.Tensor, alpha: float):
        chr_ = CHRCalibrate(self.quantiles.squeeze().cpu().numpy(), randomize=False)
        q_calib = self(x_val.to(self.device)).cpu().numpy()
        chr_.calibrate(q_calib=q_calib, Y=y_val.squeeze().cpu().numpy(), alpha=alpha)
        return chr_

    @torch.no_grad()
    def get_metrics(
        self, x_test: torch.Tensor, y_test: torch.Tensor, chr: CHRCalibrate,
    ) -> dict[str, float]:
        q_new = self(x_test.to(self.device)).cpu().numpy()
        bands = chr.predict(q_new=q_new)
        y = y_test.squeeze().numpy()
        covered = ((y >= bands[:, 0]) & (y <= bands[:, 1]))
        sizes = torch.tensor(bands[:, 1] - bands[:, 0], dtype=y_test.dtype)
        return compute_conformal_metrics(x_test, y_test, sizes=sizes, covered=torch.tensor(covered))


def plot_chr(
    model: CHR, x: torch.Tensor, chr_: CHRCalibrate,
    y: float = None, ax: plt.Axes = None,
):
    x = x.to(model.device)
    with torch.no_grad():
        q_new = model(x).cpu().numpy()
    left, right = chr_.predict(q_new=q_new)[0]
    quantiles = model.quantiles.cpu().squeeze()
    # ax.plot(quantiles, q_new.squeeze())
    q_new = q_new.squeeze()
    ax.plot(q_new.squeeze(), quantiles, c="k")
    ax.set_ylabel("$\\hatF_{\\theta}(y \\mid x)$")
    # density = (quantiles[1:] - quantiles[:-1]) / (q_new[1:] - q_new[:-1])
    # ax.plot(q_new[:-1], density, c="k", label="PDF")

    # predictive interval
    if y is not None:
        y_covered = (y >= left) & (y <= right)
        likelihood_color = "green" if y_covered else "red"
        ax.axvline(y, color=likelihood_color, ls="--", label="true $y$")
    else:
        likelihood_color = "green"
    ymin, ymax = ax.get_ylim()
    x = np.arange(left, right, 0.01)
    y1 = np.repeat(0, len(x))
    y2 = np.repeat(ymax, len(x))
    size = right - left
    ax.fill_between(x, y1, y2, color=likelihood_color, alpha=0.1, label=f"$|\\mathcal{{C}}(x)| = {size:.2f}$")
    ax.set_xlabel("$y$")
    ax.legend()
