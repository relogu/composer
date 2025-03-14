# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for optimizers and metric calculation."""

from typing import Callable

import torch
from torch import Tensor

curvature_names = {
    'local_lipschitz': 'local_lipschitz',
    'long_bb': 'long_bb',
    'short_bb': 'short_bb',
    'l2_norm/first_to_second_derivative_estimate_ratio': 'l2_norm/first_to_second_derivative_estimate_ratio',
    'l2_norm/second_derivative_estimate': 'l2_norm/second_derivative_estimate',
}


def get_report_curvature() -> Callable[[Tensor, str], dict[str, Tensor]]:
    """Get the report curvature function closure.

    Returns:
    Callable[[Any, Any, Any], float]: The report curvature function.

    Reference: AdaBB: Adaptive Barzilai-Borwein Method for Convex Optimization
    page 2, equations 3,4,5
    """
    prev_grad: Tensor | None = None
    prev_param: Tensor | None = None

    def report_curvature(param: Tensor, name: str) -> dict[str, Tensor]:
        nonlocal prev_grad, prev_param
        if param.grad is None:
            # Nothing to report
            return {}
        if prev_grad is None or prev_param is None:
            prev_grad = param.grad.detach().clone()
            prev_param = param.detach().clone()
            return {}

        grad_diff = param.grad - prev_grad
        param_diff = param - prev_param

        param_diff_norm: Tensor = torch.linalg.vector_norm(param_diff)
        grad_diff_norm: Tensor = torch.linalg.vector_norm(grad_diff)
        long_bb: Tensor = param_diff_norm**2.0 / torch.mul(grad_diff, param_diff)
        short_bb: Tensor = torch.mul(grad_diff, param_diff) / grad_diff_norm**2.0
        second_derivative_estimate: Tensor = grad_diff / param_diff
        first_to_second_derivative_ratio: Tensor = torch.linalg.vector_norm(param.grad / second_derivative_estimate)
        local_lipschitz: Tensor = grad_diff_norm / param_diff_norm
        prev_grad = param.grad.detach().clone()
        prev_param = param.detach().clone()
        return {
            f'curvature/param_diff_norm/{name}':
                local_lipschitz,
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
