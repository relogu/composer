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
import warnings
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
        # Initialize the store of gradients accumulated
        self.last_gradients_store: dict[str, torch.Tensor] = {}
        self.g_small_layer_l2norm_squared: dict[str, torch.Tensor] = {}
        self.g_small_l2norm_squared = torch.tensor(0.0)
        self.g_big_l2norm_squared = torch.tensor(0.0)

    def after_backward(self, state: State, logger: Logger) -> None:
        self.g_small_l2norm_squared = torch.tensor(0.0)
        self.g_big_l2norm_squared = torch.tensor(0.0)
        # NOTE: This event is triggered after the `loss.backward()` call of a microbatch has been completed. Depending on the sharding strategy and its underlying synchronization mechanism, the gradients contained in the `p.grad` variables (`for _name, p in state.model.named_parameters()`) could have been already synchronized across all ranks or not. We assume here that the gradients are synchronized across all ranks every microbatch step, which is the default behavior, see Trainer._train_microbatches() function. When this assumption hold true, at this stage of the pipeline each rank has the same gradients.
        if self.is_first_step:
            warnings.warn(
                'NoiseScaleMonitor assumes that the gradients are synchronized across all ranks every microbatch step when the `Event.AFTER_BACKWARD` event is triggered. If this is not the case, the noise scale estimation could be incorrect.',
                category=UserWarning,
            )
            self.is_first_step = False
        assert state.device_train_microbatch_size is not None, "The microbatch size must be set"
        # NOTE: $|G_{small}|^2$ is estimated by taking the average of the squared L2-norms of the gradients increments (exposed during every pass of the `Event.AFTER_BACKWARD` event), which is computed by subtracting the gradient from the `p.grad` variables (`for _name, p in state.model.named_parameters()`) the value at the previous step.
        # NOTE: $|G_{big}|^2$ is estimated by taking the squared L2-norm of the gradients exposed during the last pass of the `Event.AFTER_BACKWARD` event. In fact, the the `p.grad` variables (`for _name, p in state.model.named_parameters()`) contain the average of all microbatch gradients across all ranks across all microbatches.
        # NOTE: When the training doesn't take advantage any distributed environment, the `loss.backward()` call accumulates gradients by summing and not averaging, unlike the equivalent call in a distributed environment. As such, the `p.grad` variables contain the sum of all microbatch gradients across all microbatches. While this may sound to be a problem in our computation, ti is actually accounted for when computing the $B_{small}$ and the $B_big$ batch sizes in the trace estimator and the squared gradients estimator.
        # Increment the number of microbatches seen (number of accumulation steps).
        self.num_microbatches += 1
        # Loop over gradient vectors and update the accumulators
        current_gradient_increment: dict[str, torch.Tensor] = {}
        grads_small: list[torch.Tensor] = []
        grads_big: list[torch.Tensor] = []
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                grads_small.append(p.grad.flatten().view(-1, 1).detach().clone())
                grads_big.append(p.grad.flatten().view(-1, 1).detach().clone())
        
        grad_small = torch.cat(grads_small)
        # NOTE: The current scaling factor of `grad_small` before AVG-AllReduce across ranks is not 1/self.num_microbatches, which is the correct one since it should correspond to one microbatch contribution for each microbatch step performed, but \frac{1}{n}, where $n$ is the final number of microbatch steps, which is the correct one for accumulating up until the very end. To obtain the correct scaling factor, we must multiply the `grad_small` tensor by a factor $n$//self.num_microbatches, where $n=\frac{B}{W\times b$ is the number of microbatch steps, $B$ is the global batch size, $W$ is the number of workers, and $b$ is the microbatch size. Notably, the same factor holds for correcting the AVG-AllReduced across ranks `grad_small`.
        l2_norm_grad_small = (grad_small**2).sum()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(l2_norm_grad_small)
            l2_norm_grad_small.div_(dist.get_world_size())
        self.g_small_l2norm_squared.add_(l2_norm_grad_small.to(self.g_small_l2norm_squared.device))
        
        # NOTE: When using no_sync context, the average of the `grad_big` across ranks is the correct $G_{big}$ with the correct scaling factor, \frac{b}{B}, where $b$ is the microbatch size and $B$ is the global batch size, as there are $W\times n$ contributions of weight $b$ samples, where $W$ is the number of workers and $n=\frac{B}{W\times b}$ is the number of microbatch steps.
        grad_big = torch.cat(grads_big)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad_big)
            grad_big.div_(dist.get_world_size())
        l2_norm_grad_big = (grad_big**2).sum()
        self.g_big_l2norm_squared.add_(l2_norm_grad_big.to(self.g_big_l2norm_squared.device))
        

    def batch_end(self, state: State, logger: Logger) -> None:
        # Check if we surpassed the total_steps threshold
        self.counter += 1
        if self.total_steps is not None and self.counter >= self.total_steps:
            return
        assert state.device_train_microbatch_size is not None, "The microbatch size must be set"
        log.info(f"NoiseScaleMonitor: State is using a microbatch size of {state.device_train_microbatch_size}")
        log.info(f"NoiseScaleMonitor: The number of microbatch steps executed was {self.num_microbatches}")
        
        # TODO: To be commented out when we realize whether it is necessary to unscale the gradients and how to properly do it.
        # NOTE: Change the scale of the scaler to 1.0.
        assert state.scaler is not None, "The scaler must be set"
        state.scaler.update(1.0)

        # second_g_small_l2_norm_accumulator: dict[str, torch.Tensor] = {}
        # if dist.is_initialized() and dist.get_world_size() > 1:
        #     for name in self.last_gradients_store.keys():
        #         second_g_small_l2_norm_accumulator[name] = torch.sum(self.last_gradients_store[name]**2).clone()
        #         dist.all_reduce(self.last_gradients_store[name], reduce_operation="SUM")
        #         dist.all_reduce(second_g_small_l2_norm_accumulator[name], reduce_operation="SUM")
        #         # self.last_gradients_store[name].div_(dist.get_world_size())
        #         # second_g_small_l2_norm_accumulator[name].div_(dist.get_world_size())

        # # NOTE: Loop over gradient vectors and update the store of the current gradients accumulated
        # for name, p in state.model.named_parameters():
        #     if p.grad is not None and p.requires_grad:
        #         # NOTE: We must not clip here as the tensor is scaled by the scaler. The clipping, if applied, must act on the unscaled tensors.
        #         if name in self.last_gradients_store:
        #             self.last_gradients_store[name].mul_(.0)
        #             self.last_gradients_store[name].add_(p.grad.flatten().view(-1, 1).clone())
        #         else:
        #             self.last_gradients_store[name] = p.grad.flatten().view(-1, 1).clone()

        # # Put accumulators in a list for easier manipulation
        # last_gradients_store_list = [g for g in self.last_gradients_store.values()]
        # g_small_layer_l2norm_squared_list = [g for g in self.g_small_layer_l2norm_squared.values()]
        # clip_tensor_norm_(
        #     g_small_layer_l2norm_squared_list,
        #     1.0,
        # )

        # # NOTE: Unscale the gradients of the big batch
        # assert state.scaler is not None, "The scaler must be set"
        # log.info(f"NoiseScaleMonitor: State is using a {type(state.scaler)} with scale={state.scaler._scale.double()}")        
        # inv_scale = state.scaler._scale.double().reciprocal().float()
        # found_inf = torch.full((), 0.0, dtype=torch.float32, device=state.scaler._scale.device)
        # found_inf_per_device = unscale_grads(
        #     last_gradients_store_list, inv_scale, found_inf, False
        # )
        # log.info(f"NoiseScaleMonitor: G_big unscaled, found_inf_per_device={found_inf_per_device}")

        # # NOTE: This clips the accumulated gradients and the single microbatch ones separately one a per-layer L2 norm of value one.
        # for g_big_accumulated_name in self.last_gradients_store.keys():
        #     clip_tensor_norm_(
        #         self.last_gradients_store[g_big_accumulated_name],
        #         1.0,
        #     )
        # for g_small_accumulated_name in self.g_small_layer_l2norm_squared.keys():
        #     clip_tensor_norm_(
        #         self.g_small_layer_l2norm_squared[g_small_accumulated_name],
        #         1.0,
        #     )

        # # Compute the L2 norms of the final version of the gradients, the that has been used by the optimizer.
        # # g_big_layer_l2norm_squared: dict[str, torch.Tensor] = {}
        # g_big_l2norm_squared: torch.Tensor = torch.tensor(0.0)
        # grads: list[torch.Tensor] = []
        # for name in self.last_gradients_store.keys():
        #     grads.append(self.last_gradients_store[name].to(g_big_l2norm_squared.device))
        #     # self.last_gradients_store[name] = self.last_gradients_store[name].to(g_big_l2norm_squared.device)
        #     # g_big_layer_l2norm_squared[f'l2_norm/grad/{name}'] = torch.sum((self.last_gradients_store[name].div(self.num_microbatches))**2)
        #     # g_big_layer_l2norm_squared[f'l2_norm/grad/{name}'] = torch.sum((self.last_gradients_store[name].div(state.device_train_microbatch_size))**2)
        #     # g_big_layer_l2norm_squared[f'l2_norm/grad/{name}'] = torch.sum((self.last_gradients_store[name].div(self.num_microbatches))**2)
        #     # g_big_l2norm_squared.add_(torch.sum((self.last_gradients_store[name].div(state.device_train_microbatch_size))**2))
        #     # g_big_l2norm_squared.add_(torch.sum((self.last_gradients_store[name])**2))
        # grad = torch.cat(grads)
        # # grad.div_(self.num_microbatches)
        # g_big_l2norm_squared.add_((grad**2).sum())

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
        
        # Sum the L2 norms of the gradients of all the layers
        # self.g_small_l2norm_squared, g_big_l2norm_squared = .0, .0
        # self.g_small_l2norm_squared.div_(self.num_microbatches)#  .div_(self.num_microbatches)
        # g_big_l2norm_squared.div_(self.num_microbatches)#.div_(self.num_microbatches)
        # for l2_grad in self.g_small_layer_l2norm_squared.values():
        #     self.g_small_l2norm_squared += float(l2_grad.div(self.num_microbatches))
        #     # self.g_small_l2norm_squared += float(l2_grad.div(state.device_train_microbatch_size))
        #     # self.g_small_l2norm_squared += float(l2_grad)
        # for l2_grad in g_big_layer_l2norm_squared.values():
        #     g_big_l2norm_squared += float(l2_grad)
        # self.g_small_l2norm_squared.mul_(self.num_microbatches**2)
        log.info("NoiseScaleMonitor: $|G_{{small}}|^2$=%s", self.g_small_l2norm_squared)
        log.info("NoiseScaleMonitor: $|G_{{big}}|^2$=%s", self.g_big_l2norm_squared)
        
        # Compute the batch sizes $B_{small}$ and $B_{big}$ as the number of microbatches times the number of workers (we assume the gradients are synchronized across all ranks every microbatch step) and $B_{small}$ times the number of microbath steps, respectively. 
        
        b_big = state.device_train_microbatch_size * self.num_microbatches
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_big *= dist.get_world_size()
        assert b_big > 0, "The batch size must be greater than 0"
        log.info("NoiseScaleMonitor: $B_{{big}}$=%s", b_big)
        
        b_small = b_big 
        if dist.is_initialized() and dist.get_world_size() > 1:
            b_small /= dist.get_world_size()
        # assert b_small > 0, "The batch size must be greater than 0"
        log.info("NoiseScaleMonitor: $B_{{small}}$=%s", b_small)
        
        # Estimate the trace of the covariance matrix of the gradients
        trace_estimate = (self.g_small_l2norm_squared.item() - self.g_big_l2norm_squared.item()) / ((1 / b_small) - (1 / b_big))
        log.info(f"NoiseScaleMonitor: trace estimate={trace_estimate}")
        # Estimate the squared norm of the gradients
        squared_gradients_estimate = (b_big * self.g_big_l2norm_squared.item() - b_small * self.g_small_l2norm_squared.item()) / (b_big - b_small)
        log.info(f"NoiseScaleMonitor: squared gradients estimate={squared_gradients_estimate}")
        
        # Compute exponential moving averages
        self.running_trace_estimate, scale = ema_with_debias(
            self.running_trace_estimate,
            self.beta,
            trace_estimate,
            self.counter
        )
        self.running_squared_gradients_estimate, noise = ema_with_debias(
            self.running_squared_gradients_estimate,
            self.beta,
            squared_gradients_estimate,
            self.counter
        )
        # Compute the noise scale
        noise_scale_ema = scale / noise
        noise_scale = trace_estimate / squared_gradients_estimate
        log.info(f"NoiseScaleMonitor: Noise Scale (EMA)={noise_scale_ema}")
        log.info(f"NoiseScaleMonitor: Noise Scale={noise_scale}")
        
        # Log the current value of the noise scale
        logger.log_metrics({"noise_scale_ema": noise_scale_ema, "noise_scale": noise_scale})

        # Reset the store of gradients accumulated
        self.last_gradients_store.clear()
        self.g_small_layer_l2norm_squared.clear()
        self.num_microbatches = 0
        self.g_small_l2norm_squared = torch.tensor(0.0)
        self.g_big_l2norm_squared = torch.tensor(0.0)
