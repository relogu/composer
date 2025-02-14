# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The module contains the NoiseScaleMonitor callback for estimating the noise scale during training.

The NoiseScaleMonitor callback estimates the noise scale of the training using unbiased estimators for the trace of the covariance matrix between samples' gradients in a batch and the L2 norm squared of the full batch gradient. This estimation method follows what is discussed in Appendix A.1 of https://arxiv.org/abs/1812.06162. The callback assumes a distributed data parallel environment using a `no_sync` context and supports single-GPU setups as well. However, it does not provide correct results when FSDP is adopted or when the default DDP context is used.

We recommend inspecting the following source files to understand the implementation details:
- composer.core.event
- composer.core.state
- composer.trainer.trainer
- composer.core.engine
- composer.distributed.dist_strategy

"""

import logging
import warnings

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ['NoiseScaleMonitor']


def ema_with_debias(avg: float | None, beta: float, y_i: float, i: int) -> tuple[float, float]:
    """Compute the exponential moving average (EMA) with debiasing.

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
    return avg, avg / (1 - beta**(i + 1))


class NoiseScaleMonitor(Callback):
    """Noise scale monitor callback.

    This callback estimates the noise scale of the training using unbiased estimators for the trace of the covariance matrix between samples' gradients in a batch and the L2 norm squared of the full batch gradient. This estimation method follows what is discussed in Appendix A.1 of https://arxiv.org/abs/1812.06162. In particular, we assume to run in a distributed data parallel environment using a `no_sync` context and to use as small batch size the device microbatch size of a single GPU. The big batch size will be the target batch size that the training configuration requires to use. Notably, this callback works even when gradient accumulation steps are performed locally, which means that the single-GPU setup is also supported. However, this callback doesn't provide the correct results when FSDP is adopted or when the default DDP context is used (because it synchronizes the gradients across workers at every `loss.backward()` call). This callback is also compatible with gradient scaling since both small gradients and big gradients are scaled by the same factor, which can be extracted from the two unbiased estimates and then canceled out during the division.

    Args:
        beta (float): The smoothing factor for the EMA. Default is 0.99.
        total_steps (int | None): The total number of steps to monitor. If None, monitor indefinitely. Default is None.
    """

    def __init__(
        self,
        beta: float = 0.99,
        total_steps: int | None = None,
    ) -> None:
        self.beta = beta
        self.total_steps = total_steps
        self.counter = 0
        self.num_microbatches = 0
        self.is_first_step = True
        # Initialize the running quantities for the exponential moving average (EMA)
        self.running_trace_estimate: float | None = None
        self.running_squared_gradients_estimate: float | None = None
        self.running_noise_scale: float | None = None
        # Initialize the store of gradients accumulated
        self.last_gradients_store: dict[str, torch.Tensor] = {}
        self.g_small_l2norm_squared: torch.Tensor | None = None

    def after_backward(self, state: State, logger: Logger) -> None:
        """Hook that is called after the backward pass.

        Args:
            state (State): The current state of training.
            logger (Logger): The logger to log metrics.
        """
        if self.is_first_step:
            warnings.warn(
                "NoiseScaleMonitor assumes that DDP or single-GPU executions are used, FSDP is not supported. It also assumes that the gradients are synchronized across all ranks just before the optimizer step is called, i.e., when the `Event.AFTER_TRAIN_BATCH` event is triggered. This can be achieved by forcing the `no_sync` context when using DDP. `no_sync` can't be used with automicrobatching. If these conditions are not satisfied, the noise scale estimation will be incorrect.",
                category=UserWarning,
            )
            self.is_first_step = False
        assert state.device_train_microbatch_size is not None, 'The microbatch size must be set'
        # Increment the number of microbatches seen (number of accumulation steps).
        self.num_microbatches += 1
        # Loop over gradient vectors and update the accumulators
        grads_small: list[torch.Tensor] = []
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                if name not in self.last_gradients_store:
                    self.last_gradients_store[name] = p.grad.flatten().detach().clone()
                    grad_increment = p.grad.flatten().detach().clone()
                else:
                    grad_increment = (
                        p.grad.flatten().detach().clone() - self.last_gradients_store[name].detach().clone()
                    )
                    self.last_gradients_store[name] = p.grad.flatten().detach().clone()
                grads_small.append(grad_increment)

        grad_small = torch.cat(grads_small)
        l2_norm_grad_small = (grad_small**2).sum()
        if self.g_small_l2norm_squared is None:
            self.g_small_l2norm_squared = l2_norm_grad_small.clone()
        else:
            self.g_small_l2norm_squared.add_(l2_norm_grad_small)

    def batch_end(self, state: State, logger: Logger) -> None:
        """Hook that is called at the end of each batch.

        Args:
            state (State): The current state of training.
            logger (Logger): The logger to log metrics.
        """
        # Check if we surpassed the total_steps threshold
        self.counter += 1
        if self.total_steps is not None and self.counter >= self.total_steps:
            return
        assert state.device_train_microbatch_size is not None, 'The microbatch size must be set'
        if self.counter == 1:
            log.info(f'NoiseScaleMonitor: State is using a microbatch size of {state.device_train_microbatch_size}')
            log.info(f'NoiseScaleMonitor: The number of microbatch steps executed was {self.num_microbatches}')

        # Compute the L2 norms of the final version of the gradients, the that has been used by the optimizer.
        g_big_l2norm_squared: torch.Tensor = torch.tensor(0.0)
        grads_big: list[torch.Tensor] = []
        for name in self.last_gradients_store.keys():
            last_grad = self.last_gradients_store[name]
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(last_grad)
                last_grad.div_(dist.get_world_size())
            grads_big.append(last_grad)
        grad_big = torch.cat(grads_big)
        l2_norm_grad_big = (grad_big**2).sum()
        g_big_l2norm_squared.add_(l2_norm_grad_big.to(g_big_l2norm_squared.device))

        # NOTE: When the small gradients, g, are computed, they have the wrong multiplicative factor: p.grad = (1 / self.num_microbatches) * g. As such, their l2 norm squared as a (1 / self.num_microbatches)**2 multiplicative factor. Since we want the average across workers and microbatch steps, the correct normalization factor is (1 / self.num_microbatches) * (1 / dist.get_world_size()) = 1 / (self.num_microbatches * dist.get_world_size()). In the following, we correct the normalization factor.
        assert self.g_small_l2norm_squared is not None, 'The small gradients must be set'
        self.g_small_l2norm_squared = self.g_small_l2norm_squared * self.num_microbatches
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.g_small_l2norm_squared)
            self.g_small_l2norm_squared = self.g_small_l2norm_squared / dist.get_world_size()

        # Compute the big and small batch sizes
        b_big = state.device_train_microbatch_size * self.num_microbatches
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_big *= dist.get_world_size()
        assert b_big > 0, 'The batch size must be greater than 0'
        b_small = state.device_train_microbatch_size

        if b_small == b_big:
            if self.counter == 1:
                warnings.warn(
                    'NoiseScaleMonitor assumes that the configured batch size is divided into microbatches. If the small and big batch sizes are equal, thish means that no microbatching is being used. The noise scale cannot be computed.',
                    category=UserWarning,
                )
        else:

            # Estimate the trace of the covariance matrix of the gradients
            trace_estimate = (self.g_small_l2norm_squared.item() - g_big_l2norm_squared.item()) / ((1 / b_small) -
                                                                                                   (1 / b_big))
            # Estimate the squared norm of the gradients
            squared_gradients_estimate = (
                b_big * g_big_l2norm_squared.item() - b_small * self.g_small_l2norm_squared.item()
            ) / (b_big - b_small)

            # Compute exponential moving averages
            self.running_trace_estimate, scale = ema_with_debias(
                self.running_trace_estimate,
                self.beta,
                trace_estimate,
                self.counter,
            )
            self.running_squared_gradients_estimate, noise = ema_with_debias(
                self.running_squared_gradients_estimate,
                self.beta,
                squared_gradients_estimate,
                self.counter,
            )
            self.running_noise_scale, noise_scale_ema_bias = ema_with_debias(
                self.running_noise_scale,
                self.beta,
                trace_estimate / squared_gradients_estimate,
                self.counter,
            )
            # Compute the noise scale
            noise_scale_with_emas = scale / noise
            noise_scale = trace_estimate / squared_gradients_estimate

            # Log the current value of the noise scale
            logger.log_metrics({
                'noise_scale/b_small': b_small,
                'noise_scale/b_big': b_big,
                'noise_scale/g_small_l2norm_squared': self.g_small_l2norm_squared.item(),
                'noise_scale/g_big_l2norm_squared': g_big_l2norm_squared.item(),
                'noise_scale/trace_estimate': trace_estimate,
                'noise_scale/squared_gradients_estimate': squared_gradients_estimate,
                'noise_scale/noise_scale_with_emas': noise_scale_with_emas,
                'noise_scale/noise_scale_ema': self.running_noise_scale,
                'noise_scale/noise_scale_ema_bias': noise_scale_ema_bias,
                'noise_scale/noise_scale_raw': noise_scale,
            },)

        # Reset the store of gradients accumulated
        self.last_gradients_store.clear()
        self.num_microbatches = 0
        self.g_small_l2norm_squared.zero_()
