# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import warnings
from typing import Any, Callable, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['OptimizerMonitor']


def accumulate_curvature_metrics(
    optimizer_metrics: dict[str, torch.Tensor],
    metric_name_prefixes: tuple[str, ...] = (
        'curvature/param_diff_norm',
        'curvature/grad_diff_norm',
        'curvature/long_bb',
        'curvature/short_bb',
        'curvature/l2_norm/first_to_second_derivative_estimate_ratio',
        'curvature/l2_norm/second_derivative_estimate',
        'curvature/local_lipschitz',
    ),
) -> dict[str, Any]:
    """Accumulate curvature metrics from the optimizer_metrics dictionary.

    Args:
        optimizer_metrics (dict[str, torch.Tensor]): The optimizer metrics dictionary.
        metric_name_prefixes (tuple[str, ...]): The prefixes of the metrics to accumulate.

    Returns:
        dict[str, Any]: The accumulated curvature metrics.
    """
    # Initialize accumulators
    sums_for_norms = {
        'curvature/param_diff_norm': 0.0,
        'curvature/grad_diff_norm': 0.0,
    }
    list_for_stats = {
        'curvature/long_bb': [],
        'curvature/short_bb': [],
        'curvature/l2_norm/first_to_second_derivative_estimate_ratio': [],
        'curvature/l2_norm/second_derivative_estimate': [],
        'curvature/local_lipschitz': [],
    }

    # Accumulate
    for metric_name, metric_value in optimizer_metrics.items():

        for prefix in metric_name_prefixes:
            if metric_name.startswith(prefix):
                # Convert any Tensor to a Python float
                val = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
                if prefix in sums_for_norms:
                    # Accumulate squared sum for these prefixes
                    sums_for_norms[prefix] += val**2
                elif prefix in list_for_stats:
                    # Just store the float
                    list_for_stats[prefix].append(val)

    return {
        'sums_for_norms': sums_for_norms,
        'list_for_stats': list_for_stats,
    }


def finalize_curvature_metrics(acc: dict[str, Any]) -> dict[str, float]:
    """Process the accumulated curvature metrics to produce final metrics.

    Args:
        acc (dict[str, Any]): The accumulated curvature metrics.

    Returns:
        dict[str, float]: The final curvature metrics.
    """
    final_metrics = {}

    # 1) Compute global norms: sqrt of the sums
    sums_for_norms = acc['sums_for_norms']
    for prefix, squared_sum in sums_for_norms.items():
        global_val = (squared_sum**0.5)
        final_metrics[f'{prefix}/global'] = global_val

    # 2) Reproduce the local_lipschitz/global calculation exactly as before
    grad_diff_g = final_metrics.get('curvature/grad_diff_norm/global', 0.0)
    param_diff_g = final_metrics.get('curvature/param_diff_norm/global', 1.0)

    final_metrics['curvature/local_lipschitz/global'] = grad_diff_g / param_diff_g if param_diff_g != 0.0 else float(
        'inf',
    )

    # 3) Compute min/mean/median/max for each metric in list_for_stats
    list_for_stats = acc['list_for_stats']
    for prefix, values in list_for_stats.items():
        if not values:
            # No values were accumulated for this prefix
            continue

        t = torch.tensor(values, dtype=torch.float32)
        # Filter out infinite values
        finite_t = t[torch.isfinite(t)]
        if finite_t.numel() == 0:
            continue

        final_metrics[f'{prefix}/mean'] = finite_t.mean().item()
        final_metrics[f'{prefix}/median'] = finite_t.median().item()
        final_metrics[f'{prefix}/min'] = finite_t.min().item()
        final_metrics[f'{prefix}/max'] = finite_t.max().item()

    return final_metrics


class OptimizerMonitor(Callback):
    """Computes and logs the L2 norm of gradients as well as any optimizer-specific metrics implemented in the optimizer's `report_per_parameter_metrics` method.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import OptimizerMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[OptimizerMonitor()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as described below. `grad_l2_norm` and `layer_grad_l2_norm` are
    logged in addition to metrics logged by the optimizer's `report_per_parameter_metrics` method. For convenience we have listed
    the metrics logged by DecoupledAdamW below.

    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    |                                               | L2 norm of the gradients of all parameters in       |
    | ``l2_norm/grad/global``                       | the model on the :attr:`.Event.AFTER_TRAIN_BATCH`   |
    |                                               | event.                                              |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms                                 |
    | ``l2_norm/grad/LAYER_NAME``                   |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam first moment after      |
    | ``l2_norm/moment/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------
    +-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of Adam second moment after      |
    | ``l2_norm/moment2/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------
    +-----------------------------------------------------+
    |                                               | Minimum element of the Adam second moment after      |
    | ``min/moment2/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------
    +-----------------------------------------------------+
    |                                               | Maximum element of the Adam second moment after      |
    | ``max/moment2/LAYER_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------
    +-----------------------------------------------------+
    |                                               | Curvature metrics derived from the gradient and model parameters      |
    | ``curvature/METRIC_NAME``                 |  calling optimizer step.                            |
    |                                               |                                                     |
    +-----------------------------------------------
    +-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of parameter weights            |
    | ``l2_norm/param/LAYER_NAME``                  |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    |                                               | Layer-wise L2 norms of the step                     |
    | ``l2_norm/update/LAYER_NAME``                 |                                                     |
    |                                               |                                                     |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(
        self,
        only_global: bool = False,
        log_optimizer_metrics: bool = True,
        interval: Union[int, str, Time] = '10ba',
        report_curvature: bool = True,
    ):
        self.log_optimizer_metrics = log_optimizer_metrics
        self.only_global = only_global

        # Check that the interval timestring is parsable and convert into time object
        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        elif isinstance(interval, str):
            self.interval = Time.from_timestring(interval)
        elif isinstance(interval, Time):
            self.interval = interval

        if self.interval.unit == TimeUnit.BATCH and self.interval < Time.from_timestring('10ba'):
            warnings.warn(
                f'Currently the ActivationMonitor`s interval is set to {self.interval} '
                f'which is below our recommended value of 10ba. We recommend you raise '
                f'the interval to at least 10ba, as the activation monitor adds extra overhead '
                f'and decreases throughput.',
            )

        # Verify that the interval has supported units
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')
        self.report_curvature = report_curvature

    def batch_end(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value != 0:
            return

        optimizer_metrics: dict = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:

                metric_reporter: Callable[[Any, Any, Any], dict] = getattr(
                    state.optimizers[0],
                    'report_per_parameter_metrics',
                    None,
                )  # type: ignore[reportAssignmentType]
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics.update(metric_reporter(p, name, optimizer_metrics))

                # Always log grad norm as a default metric if it's not specified
                if f'l2_norm/grad/{name}' not in optimizer_metrics:
                    param_grad_norm = torch.linalg.vector_norm(p.grad)
                    optimizer_metrics[f'l2_norm/grad/{name}'] = param_grad_norm

        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            # If FSDP is enabled, the optimizer state lives on different ranks and must be reduced
            # and combined before we can compute metrics.
            # Each metric has a different way of being reduced, so the optimizer is responsible for implementing
            # the reduction process.
            # It occurs first via a pre-reduce, where the metric on each rank is modified and prepared
            # then an all-reduce where the modified metric on each rank is combined into the correct metric across all ranks.
            #
            # For example, L2 norms are squared on each rank before we apply all_reduce(SUM) and take the sqrt on each rank
            pre_reduce_metrics: Callable[[Any], dict] = getattr(
                state.optimizers[0],
                'pre_reduce_metrics',
                None,
            )  # type: ignore[reportAssignmentType]
            if callable(pre_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics: Callable[[Any], dict] = getattr(
                state.optimizers[0],
                'dist_reduce_metrics',
                None,
            )  # type: ignore[reportAssignmentType]
            if callable(dist_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        grad_norm, moment_norm, moment2_norm, update_norm, param_norm = .0, .0, .0, .0, .0,
        min_moment2 = float('inf')
        max_moment2 = float('-inf')

        for metric in optimizer_metrics:
            if metric.startswith('l2_norm/grad'):
                grad_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/moment'):
                moment_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/moment2'):
                moment2_norm += optimizer_metrics[metric]**2
            if metric.startswith('min/moment2'):
                min_moment2 = min(min_moment2, optimizer_metrics[metric])
            if metric.startswith('max/moment2'):
                max_moment2 = max(max_moment2, optimizer_metrics[metric])
            if metric.startswith('l2_norm/update'):
                update_norm += optimizer_metrics[metric]**2
            if metric.startswith('l2_norm/param'):
                param_norm += optimizer_metrics[metric]**2

        curvature_acc = {}
        # Curvature metrics
        # trying to obtain them here
        # has minimal cost if the optimizer
        # does not report them
        if self.report_curvature:
            curvature_acc = accumulate_curvature_metrics(optimizer_metrics)

        if self.only_global:
            optimizer_metrics = {}

        optimizer_metrics['l2_norm/grad/global'] = grad_norm**0.5
        optimizer_metrics['l2_norm/moment/global'] = moment_norm**0.5
        optimizer_metrics['l2_norm/moment2/global'] = moment2_norm**0.5
        optimizer_metrics['l2_norm/update/global'] = update_norm**0.5
        optimizer_metrics['l2_norm/param/global'] = param_norm**0.5
        optimizer_metrics['min/moment2/global'] = min_moment2
        optimizer_metrics['max/moment2/global'] = max_moment2

        if self.report_curvature:
            curvature_stats = finalize_curvature_metrics(curvature_acc)
            optimizer_metrics.update(curvature_stats)

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log_metrics(optimizer_metrics)
