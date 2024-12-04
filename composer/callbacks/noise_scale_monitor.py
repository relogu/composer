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
from collections import defaultdict
from collections.abc import Iterable
import logging
import numpy as np
import torch
from torch.nn.utils.clip_grad import _tensor_or_tensors
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from torch.amp.grad_scaler import _MultiDeviceReplicator

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ['NoiseScaleMonitor']

# Freely inspired from torch.nn.utils.clip_grad_norm_
def clip_tensor_norm_(
    input_parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    parameters: Iterable[torch.Tensor]
    if isinstance(input_parameters, torch.Tensor):
        parameters = [input_parameters]
    else:
        parameters = input_parameters
    # NOTE: In this implementation, the input parameters are already the gradients
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:  # type: ignore[reportArgumentType]
        return torch.tensor(0.0)
    first_device = parameters[0].device  # type: ignore[reportIndexIssue]
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[torch.Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [parameters] # type: ignore[assignment]
    )  

    norms: list[torch.Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm

def unscale_grads(
    grads: list[torch.Tensor],
    inv_scale: torch.Tensor,
    found_inf: torch.Tensor,
    allow_fp16: bool,
) -> dict[torch.device, torch.Tensor]:
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)

    # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
    # There could be hundreds of grads, so we'd like to iterate through them just once.
    # However, we don't know their devices or dtypes in advance.

    # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
    # Google says mypy struggles with defaultdicts type annotations.
    per_device_and_dtype_grads: dict[
        torch.device, dict[torch.dtype, list[torch.Tensor]]
    ] = defaultdict(lambda: defaultdict(list))
    for grad in grads:
        assert isinstance(grad, torch.Tensor)
        if grad is None:
            continue
        if (not allow_fp16) and grad.dtype == torch.float16:
            raise ValueError("Attempting to unscale FP16 gradients.")
        if grad.is_sparse:
            # is_coalesced() == False means the sparse grad has values with duplicate indices.
            # coalesce() deduplicates indices and adds all values that have the same index.
            # For scaled fp16 values, there's a good chance coalescing will cause overflow,
            # so we should check the coalesced _values().
            if grad.dtype is torch.float16:
                grad = grad.coalesce()
            to_unscale = grad._values()
        else:
            to_unscale = grad

        # TODO: is there a way to split by device and dtype without appending in the inner loop?
        per_device_and_dtype_grads[to_unscale.device][
            to_unscale.dtype
        ].append(to_unscale)

    for device, per_dtype_grads in per_device_and_dtype_grads.items():
        for grads_to_scale in per_dtype_grads.values():
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads_to_scale,
                per_device_found_inf.get(device),
                per_device_inv_scale.get(device),
            )

    return per_device_found_inf._per_device_tensors


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
        max_norm: float = 1.0,
        beta: float = 0.99,
        total_steps: int | None = None,
    ) -> None:
        self.estimation_window = estimation_window
        self.max_norm = max_norm
        self.beta = beta
        self.total_steps = total_steps
        self.counter = 0
        self.num_microbatches = 0
        # Initialize the running quantities for the exponential moving average (EMA)
        self.running_trace_estimate1: float | None = None
        self.running_squared_gradients_estimate1: float | None = None
        self.running_trace_estimate2: float | None = None
        self.running_squared_gradients_estimate2: float | None = None
        # Initialize the store of gradients accumulated
        self.g_big_accumulator: dict[str, torch.Tensor] = {}
        self.g_small_accumulator: dict[str, torch.Tensor] = {}
        self.g_small_l2_norm_accumulator: dict[str, torch.Tensor] = {}
        self.g_big: torch.Tensor | None = None

    def after_backward(self, state: State, logger: Logger) -> None:
        # NOTE: This event is triggered after the `loss.backward()` call of a microbatch has been completed. As such, the gradients on the parameter contain the accumulated gradients across the microbatches executed after the last optimizer step. The first call of this event after the optimizer step is the first time we are exposed to the G_small, i.e., gradients of a single microbatch. No scaling or clipping is performed on these gradients.
        # log.info("NoiseScaleMonitor: after backward pass on a microbatch")
        self.num_microbatches += 1
        # Loop over gradient vectors and update the accumulators
        current_g_small: dict[str, torch.Tensor] = {}
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                if name in self.g_big_accumulator:
                    # NOTE: The current gradient contains the average of all the microbatch gradients accumulated so far.
                    # To extract the gradient of the current microbatch, we multiply by the number of microbatches and subtract the accumulated gradients multiplied by their scaling factor (`self.num_microbatches - 1`).
                    # current_g_small[name] = (
                    #     p.grad.flatten().view(-1, 1).clone().mul(self.num_microbatches) - \
                    #     self.g_big_accumulator[name].mul(self.num_microbatches - 1)
                    # )
                    current_g_small[name] = (
                        p.grad.flatten().view(-1, 1).clone() - \
                        self.g_big_accumulator[name]
                    )
                    # Then, we assign the latest accumulated gradient to the accumulator
                    self.g_big_accumulator[name].mul_(.0)
                    self.g_big_accumulator[name].add_(p.grad.flatten().view(-1, 1).clone())
                else:
                    # NOTE: This is the fist time we are exposed to microbatch gradients. As such, we initialize the accumulator with the current microbatch gradient.
                    self.g_big_accumulator[name] = p.grad.flatten().view(-1, 1).clone()
                    current_g_small[name] = p.grad.flatten().view(-1, 1).clone()
        # NOTE: This won't work for FSDP
        # Clip norm of the gradients of the current microbatch
        if dist.is_initialized() and dist.get_world_size() > 1:
            for name in current_g_small.keys():
                dist.all_reduce(current_g_small[name], reduce_operation="SUM")
                # current_g_small[name].div_(dist.get_world_size())
        # current_g_small_list = [g for g in current_g_small.values()]
        # clip_tensor_norm_(
        #     current_g_small_list,
        #     1.0,
        # )
        for name in current_g_small.keys():
            if f'l2_norm/grad/{name}' in self.g_small_accumulator:
                # NOTE: The accumulator contains the average of the L2 norm of the gradients.
                # First, we scale it with the scaling factor of the number of microbatch gradients it contains (`self.num_microbatches - 1`)
                # self.g_small_accumulator[f'l2_norm/grad/{name}'].mul_(self.num_microbatches - 1)
                # Second, we add the L2 norm of the current microbatch gradient
                self.g_small_accumulator[f'l2_norm/grad/{name}'].add_(torch.sum(current_g_small[name]**2))
                # Third, we divide by the number of microbatches it now contains
                # self.g_small_accumulator[f'l2_norm/grad/{name}'].div_(self.num_microbatches)
            else:
                # NOTE: This is the fist time we are exposed to microbatch gradients. As such, we use their L2 norm to initialize the G_small accumulator. We dont't need to square it as we will square it at the next iteration before the actual summation happens.
                self.g_small_accumulator[f'l2_norm/grad/{name}'] = torch.sum(current_g_small[name]**2)

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to update and log the noise scale.

        Args:
            state (State): The current state of training.
            logger (Logger): The logger to log metrics.
        """
        # Check if we surpassed the total_steps threshold
        self.counter += 1
        if self.total_steps is not None and self.counter >= self.total_steps:
            return
        log.info("NoiseScaleMonitor: all microbatches gradients have been accumulated and the optimizer step performed")

        second_g_small_l2_norm_accumulator: dict[str, torch.Tensor] = {}
        if dist.is_initialized() and dist.get_world_size() > 1:
            for name in self.g_big_accumulator.keys():
                second_g_small_l2_norm_accumulator[name] = torch.sum(self.g_big_accumulator[name]**2).clone()
                dist.all_reduce(self.g_big_accumulator[name], reduce_operation="SUM")
                dist.all_reduce(second_g_small_l2_norm_accumulator[name], reduce_operation="SUM")
                # self.g_big_accumulator[name].div_(dist.get_world_size())
                # second_g_small_l2_norm_accumulator[name].div_(dist.get_world_size())

        # # NOTE: Loop over gradient vectors and update the store of the current gradients accumulated
        # for name, p in state.model.named_parameters():
        #     if p.grad is not None and p.requires_grad:
        #         # NOTE: We must not clip here as the tensor is scaled by the scaler. The clipping, if applied, must act on the unscaled tensors.
        #         if name in self.g_big_accumulator:
        #             self.g_big_accumulator[name].mul_(.0)
        #             self.g_big_accumulator[name].add_(p.grad.flatten().view(-1, 1).clone())
        #         else:
        #             self.g_big_accumulator[name] = p.grad.flatten().view(-1, 1).clone()

        # # Put accumulators in a list for easier manipulation
        # g_big_accumulator_list = [g for g in self.g_big_accumulator.values()]
        # g_small_accumulator_list = [g for g in self.g_small_accumulator.values()]
        # clip_tensor_norm_(
        #     g_small_accumulator_list,
        #     1.0,
        # )
        
        # NOTE Change the scale of the scaler to 1.0
        assert state.scaler is not None, "The scaler must be set"
        state.scaler.update(1.0)

        # # NOTE: Unscale the gradients of the big batch
        # assert state.scaler is not None, "The scaler must be set"
        # log.info(f"NoiseScaleMonitor: State is using a {type(state.scaler)} with scale={state.scaler._scale.double()}")        
        # inv_scale = state.scaler._scale.double().reciprocal().float()
        # found_inf = torch.full((), 0.0, dtype=torch.float32, device=state.scaler._scale.device)
        # found_inf_per_device = unscale_grads(
        #     g_big_accumulator_list, inv_scale, found_inf, False
        # )
        # log.info(f"NoiseScaleMonitor: G_big unscaled, found_inf_per_device={found_inf_per_device}")

        # # NOTE: This clips the accumulated gradients and the single microbatch ones separately one a per-layer L2 norm of value one.
        # for g_big_accumulated_name in self.g_big_accumulator.keys():
        #     clip_tensor_norm_(
        #         self.g_big_accumulator[g_big_accumulated_name],
        #         1.0,
        #     )
        # for g_small_accumulated_name in self.g_small_accumulator.keys():
        #     clip_tensor_norm_(
        #         self.g_small_accumulator[g_small_accumulated_name],
        #         1.0,
        #     )

        # Loop over gradient vectors and update the store of the current gradients accumulated
        # AllReduce the big gradients and the small gradients across all ranks
        g_big_reduced: dict[str, torch.Tensor] = {}
        for name in self.g_big_accumulator.keys():
            g_big_reduced[f'l2_norm/grad/{name}'] = torch.sum(self.g_big_accumulator[name]**2)
        g_small_reduced1 = self.g_small_accumulator
        g_small_reduced2 = second_g_small_l2_norm_accumulator

        # # NOTE: If FSDP is enabled, the optimizer states may live on different ranks and must be reduced accordingly
        # if state.fsdp_enabled and dist.get_world_size() > 0:
        #     # If FSDP is enabled, the optimizer state lives on different ranks and must be reduced
        #     # and combined before we can compute metrics.
        #     # Each metric has a different way of being reduced, so the optimizer is responsible for implementing
        #     # the reduction process.
        #     # It occurs first via a pre-reduce, where the metric on each rank is modified and prepared
        #     # then an all-reduce where the modified metric on each rank is combined into the correct metric across all ranks.
        #     # For example, L2 norms are squared on each rank before we apply all_reduce(SUM) and take the sqrt on each rank
        #     pre_reduce_metrics = getattr(state.optimizers[0], 'pre_reduce_metrics', None)
        #     if callable(pre_reduce_metrics):
        #         g_big_reduced = pre_reduce_metrics(g_big_reduced)
        #         g_small_reduced = pre_reduce_metrics(g_small_reduced)

        #     dist_reduce_metrics = getattr(state.optimizers[0], 'dist_reduce_metrics', None)
        #     if callable(dist_reduce_metrics):
        #         g_big_reduced = dist_reduce_metrics(g_big_reduced)
        #         g_small_reduced = dist_reduce_metrics(g_small_reduced)
        
        # Compute the sum of the clipped gradient
        g_small_norm_sq1, g_small_norm_sq2, g_big_norm_sq = .0, .0, .0
        for l2_grad in g_small_reduced1.values():
            g_small_norm_sq1 += float(l2_grad)
        for l2_grad in g_small_reduced2.values():
            g_small_norm_sq2 += float(l2_grad)
        for l2_grad in g_big_reduced.values():
            g_big_norm_sq += float(l2_grad)
        log.info(f"NoiseScaleMonitor: |G_small1|^2={g_small_norm_sq1}")
        log.info(f"NoiseScaleMonitor: |G_small2|^2={g_small_norm_sq2}")
        log.info(f"NoiseScaleMonitor: |G_big|^2={g_big_norm_sq}")
        
        # The big batch size is the product of the small batch size and the estimation window
        assert state.device_train_microbatch_size is not None, "The microbatch size must be set"
        b_small1 = state.device_train_microbatch_size
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_small1 *= dist.get_world_size()
        assert b_small1 > 0, "The batch size must be greater than 0"
        log.info(f"NoiseScaleMonitor: batch size small1={b_small1}")
        # Get small batch size from the State object, i.e., the actual batch size the training pipeline uses
        b_big = state.device_train_microbatch_size * self.num_microbatches
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_big *= dist.get_world_size()
        assert b_big > 0, "The batch size must be greater than 0"
        log.info(f"NoiseScaleMonitor: batch size big={b_big}")
        b_small2: float | None = None
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_small2 = b_big / dist.get_world_size()
        log.info(f"NoiseScaleMonitor: batch size small2={b_small2}")
        # Estimate the trace of the covariance matrix of the gradients
        trace_estimate1 = ((1 / ((1 / b_small1) - (1 / b_big))) * (g_small_norm_sq1 - g_big_norm_sq))
        log.info(f"NoiseScaleMonitor: trace estimate1={trace_estimate1}")
        # Estimate the squared norm of the gradients
        squared_gradients_estimate1 = ((1 / (b_big - b_small1)) * (b_big * g_big_norm_sq - b_small1 * g_small_norm_sq1))
        log.info(f"NoiseScaleMonitor: squared gradients estimate1={squared_gradients_estimate1}")
        # Compute exponential moving averages
        self.running_trace_estimate1, scale1 = ema_with_debias(
            self.running_trace_estimate1,
            self.beta,
            trace_estimate1,
            self.counter // self.estimation_window
        )
        self.running_squared_gradients_estimate1, noise1 = ema_with_debias(
            self.running_squared_gradients_estimate1,
            self.beta,
            squared_gradients_estimate1,
            self.counter // self.estimation_window
        )
        # Compute the noise scale
        noise_scale_ema1 = scale1 / noise1
        noise_scale1 = trace_estimate1 / squared_gradients_estimate1
        
        noise_scale_ema2 = -1.0
        noise_scale2 = -1.0
        if b_small2 is not None:
            # Estimate the trace of the covariance matrix of the gradients
            trace_estimate2 = ((1 / ((1 / b_small2) - (1 / b_big))) * (g_small_norm_sq2 - g_big_norm_sq))
            log.info(f"NoiseScaleMonitor: trace estimate2={trace_estimate2}")
            # Estimate the squared norm of the gradients
            squared_gradients_estimate2 = ((1 / (b_big - b_small2)) * (b_big * g_big_norm_sq - b_small2 * g_small_norm_sq2))
            log.info(f"NoiseScaleMonitor: squared gradients estimate2={squared_gradients_estimate2}")
            # Compute exponential moving averages
            self.running_trace_estimate2, scale2 = ema_with_debias(
                self.running_trace_estimate2,
                self.beta,
                trace_estimate2,
                self.counter // self.estimation_window
            )
            self.running_squared_gradients_estimate2, noise2 = ema_with_debias(
                self.running_squared_gradients_estimate2,
                self.beta,
                squared_gradients_estimate2,
                self.counter // self.estimation_window
            )
            # Compute the noise scale
            noise_scale_ema2 = scale2 / noise2
            noise_scale2 = trace_estimate2 / squared_gradients_estimate2
        
        # Log the current value of the noise scale
        logger.log_metrics({"noise_scale_ema1": noise_scale_ema1, "noise_scale1": noise_scale1, "noise_scale_ema2": noise_scale_ema2, "noise_scale2": noise_scale2})
        # Reset the store of gradients accumulated
        for name in self.g_big_accumulator.keys():
            self.g_big_accumulator[name].zero_()
        self.g_small_accumulator.clear()
        self.num_microbatches = 0