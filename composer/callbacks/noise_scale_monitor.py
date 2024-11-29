"""
This module contains the NoiseScaleMonitor class, a callback for monitoring and logging the noise scale during training.
It also includes a helper function for computing the exponential moving average (EMA) with debiasing.
Inspired by https://arxiv.org/abs/1812.06162 and https://github.com/hal-314/fastai-batch-size-finder.
Author: Lorenzo Sani <ls985@cam.ac.uk>

Classes:
    NoiseScaleMonitor: A callback to monitor and log the noise scale during training.

Functions:
    ema_with_debias: Compute the exponential moving average (EMA) with debiasing.
"""
import logging
import torch

from composer.core import Callback, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['NoiseScaleMonitor']


def ema_with_debias(avg: float | None, beta: float, y_i: float, i: int) -> tuple[float, float]:
    """
    Compute the exponential moving average (EMA) with debiasing.

    Args:
        avg (float | None): The current average value.
        beta (float): The smoothing factor for the EMA.
        y_i (float): The new value to include in the EMA.
        i (int): The current step count.

    Returns:
        tuple[float, float]: The updated average and the debiased average.
    """
    if avg is None:
        avg = 0.0
    avg = beta * avg + (1 - beta) * y_i
    return avg, avg / (1 - beta ** (i + 1))


class NoiseScaleMonitor(Callback):
    """
    A callback to monitor and log the noise scale during training.

    Args:
        estimation_window (int): The window size for noise scale estimation. Defaults to 5.
        beta (float): The smoothing factor for the exponential moving average. Defaults to 0.99.
        total_steps (int | None): The total number of steps to monitor. If None, monitor indefinitely. Defaults to None.
    """

    def __init__(
        self,
        estimation_window: int = 5,
        beta: float = 0.99,
        total_steps: int | None = None,
    ) -> None:
        self.estimation_window = estimation_window
        self.beta = beta
        self.total_steps = total_steps
        self.counter = 0
        # Initialize the running quantities for the exponential moving average (EMA)
        self.running_trace_estimate: float | None = None
        self.running_squared_gradients_estimate: float | None = None
        # Initialize the store of gradients accumulated
        self.g_big: torch.Tensor = torch.tensor([])

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to update and log the noise scale.

        Args:
            state (State): The current state of training.
            logger (Logger): The logger to log metrics.
        """
        # Check if we surpassed the total_steps threshold
        if self.total_steps is not None and self.counter >= self.total_steps:
            return
        self.counter += 1
        # Loop over gradient vectors and update the store of the current gradients accumulated
        current_gradients: list[torch.Tensor] = []
        for _name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                # Collect the current gradient vector into the list of the current gradients. Make each gradient a rank-1 tensor.
                current_gradients.append(p.grad.flatten().view(-1, 1).clone())
        # Append the current gradients as a single vector to the gradient accumulator
        g_small = torch.cat(current_gradients)
        if self.g_big.numel() == 0:
            self.g_big = g_small.clone()
        else:
            self.g_big += g_small.clone()
        # If we reached the window, compute the noise scale
        if self.counter % self.estimation_window == 0:
            # Get small batch size from the State object, i.e., the actual batch size the training pipeline uses
            b_small = len(state.batch)
            assert b_small > 0, "The batch size must be greater than 0"
            # The big batch size is the product of the small batch size and the estimation window
            b_big = b_small * self.estimation_window
            # Estimate the trace of the covariance matrix of the gradients
            trace_estimate = ((1 / ((1 / b_small) - (1 / b_big))) * (torch.linalg.norm(g_small) ** 2 - torch.linalg.norm(self.g_big) ** 2)).item()
            # Estimate the squared norm of the gradients
            squared_gradients_estimate = ((1 / (b_big - b_small)) * (b_big * torch.linalg.norm(self.g_big) ** 2 - b_small * torch.linalg.norm(g_small) ** 2)).item()
            # Compute exponential moving averages
            self.running_trace_estimate, scale = ema_with_debias(
                self.running_trace_estimate,
                self.beta,
                trace_estimate,
                self.counter // self.estimation_window
            )
            self.running_squared_gradients_estimate, noise = ema_with_debias(
                self.running_squared_gradients_estimate,
                self.beta,
                squared_gradients_estimate,
                self.counter // self.estimation_window
            )
            # Compute the noise scale
            noise_scale = scale / noise
            # Log the current value of the noise scale
            logger.log_metrics({"noise_scale": noise_scale})