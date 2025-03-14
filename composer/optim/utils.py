# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for optimizers and metric calculation."""

from typing import Callable

import torch
from torch import Tensor


def get_report_curvature() -> Callable[[Tensor, str], dict[str, Tensor]]:
    """Get a function that computes curvature metrics per-parameter.

    Returns:
        Callable[[Tensor, str], dict[str, Tensor]]: A function that, given a
        parameter `param` and its string name `name`, returns a dictionary with
        curvature metrics for that parameter. The function internally tracks
        previous parameter and gradient values for each named parameter.
    """
    prev_params: dict[str, Tensor] = {}
    prev_grads: dict[str, Tensor] = {}

    def report_curvature(param: Tensor, name: str) -> dict[str, Tensor]:
        # If there's no gradient, we skip reporting
        if param.grad is None:
            return {}

        # If we've never seen this parameter before, initialize storage and skip
        # this round (can't compute diffs without history).
        if name not in prev_params or name not in prev_grads:
            prev_params[name] = param.detach().clone()
            prev_grads[name] = param.grad.detach().clone()
            return {}

        # Compute diffs
        grad_diff = param.grad - prev_grads[name]
        param_diff = param - prev_params[name]

        param_diff_norm: Tensor = torch.linalg.vector_norm(param_diff)
        grad_diff_norm: Tensor = torch.linalg.vector_norm(grad_diff)

        # Barzilai–Borwein "long" step size
        long_bb = param_diff_norm**2.0 / torch.mul(grad_diff, param_diff)

        # Barzilai–Borwein "short" step size
        short_bb = torch.mul(grad_diff, param_diff) / grad_diff_norm**2.0

        # Second-derivative estimate (elementwise ratio)
        second_derivative_estimate = grad_diff / param_diff

        # Ratio of first to second derivative
        first_to_second_derivative_ratio = torch.linalg.vector_norm(param.grad / second_derivative_estimate,)

        # Update stored values for next iteration
        prev_params[name] = param.detach().clone()
        prev_grads[name] = param.grad.detach().clone()

        return {
            f'curvature/param_diff_norm/{name}':
                param_diff_norm,
            f'curvature/grad_diff_norm/{name}':
                grad_diff_norm,
            f'curvature/long_bb/{name}':
                long_bb,
            f'curvature/short_bb/{name}':
                short_bb,
            f'curvature/l2_norm/first_to_second_derivative_estimate_ratio/{name}':
                first_to_second_derivative_ratio,
            f'curvature/l2_norm/second_derivative_estimate/{name}':
                torch.linalg.vector_norm(second_derivative_estimate),
        }

    return report_curvature
