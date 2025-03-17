# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for optimizers and metric calculation.

Currently handles curvature metrics shared across ADAM-like optimizers.
"""

from typing import Callable

import torch
from torch import Tensor


def get_report_curvature() -> Callable[[Tensor, str], dict[str, Tensor]]:
    """Get a function that computes curvature metrics per-parameter.

    The function tracks differences in parameters and gradients between iterations, thus it allocates extra memory. To avoid interfering with training, it saves the previous parameter and gradient values for each named parameter on the cpu, and only sends them to the device when needed for calculation, for 2 H100s this causes
    a drop in average samples per second from ~96 to ~94.

    Metrics are primarily taken from AdaBB: Adaptive Barzilai-Borwein Method for Convex Optimization: https://arxiv.org/pdf/2401.08024
    specifically equations 3,4,5 from page 2. Metrics include:
    - param_diff_norm: L2 norm of the parameter difference between iterations
    - grad_diff_norm: L2 norm of the gradient difference between iterations
    - local_lipschitz: Calculated later, it is the ratio of the gradient difference norm to the parameter difference norm and estimates
    the local Lipschitz constant. Its supposed to capture the local curvature of the loss function (https://arxiv.org/pdf/2401.08024)
    - second_derivative_estimate: L2 norm of the elementwise ratio of gradient difference to parameter difference, recommended by a collaborator
    - second_to_first_derivative_estimate_ratio: L2 norm of the elementwise ratio of the second derivative estimate to the first derivative, recommended by a collaborator
    - long_bb: Barzilai-Borwein "long" step size (equation 3 from https://arxiv.org/pdf/2401.08024)
    - short_bb: Barzilai-Borwein (https://arxiv.org/pdf/2401.08024) "short" step size (equation 4 from https://arxiv.org/pdf/2401.08024)





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
            prev_params[name] = param.detach().clone().cpu()
            prev_grads[name] = param.grad.detach().clone().cpu()
            return {}

        # Compute diffs
        grad_diff = param.grad - prev_grads[name].to(device=param.grad.device, dtype=param.grad.dtype)
        param_diff = param - prev_params[name].to(device=param.device, dtype=param.dtype)

        param_diff_norm: Tensor = torch.linalg.vector_norm(param_diff)
        grad_diff_norm: Tensor = torch.linalg.vector_norm(grad_diff)

        # Barzilai–Borwein "long" step size
        long_bb = param_diff_norm**2.0 / torch.sum(torch.mul(grad_diff, param_diff))

        # Barzilai–Borwein "short" step size
        short_bb = torch.sum(torch.mul(grad_diff, param_diff)) / grad_diff_norm**2.0

        # Second-derivative estimate (elementwise ratio)
        second_derivative_estimate = grad_diff / param_diff

        # Ratio of first to second derivative
        second_to_first_derivative_ratio = torch.linalg.vector_norm(second_derivative_estimate / param.grad)

        # Update stored values for next iteration
        prev_params[name] = param.detach().clone().cpu()
        prev_grads[name] = param.grad.detach().clone().cpu()

        return {
            f'curvature/param_diff_norm/{name}':
                param_diff_norm,
            f'curvature/grad_diff_norm/{name}':
                grad_diff_norm,
            f'curvature/long_bb/{name}':
                long_bb,
            f'curvature/short_bb/{name}':
                short_bb,
            f'curvature/l2_norm/second_to_first_derivative_estimate_ratio/{name}':
                second_to_first_derivative_ratio,
            f'curvature/l2_norm/second_derivative_estimate/{name}':
                torch.linalg.vector_norm(second_derivative_estimate),
        }

    return report_curvature
