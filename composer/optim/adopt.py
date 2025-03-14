# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

r"""Implementation of the ADOPT optimizer.

Paper: https://arxiv.org/abs/2411.02853
Original code: https://github.com/iShohei220/adopt

ADOPT (Adam with Optimal Rate for any \beta_2) is a gradient-based optimizer
designed to fix the convergence of Adam while introducing an adaptive
clipping step to stabilize updates and handle large gradients effectively.

This module provides both the high-level :class:`ADOPT` class, an `Optimizer`
subclass suitable for direct usage, and the low-level functional API
:func:`adopt` that performs the same computations. Typically, users will
instantiate the :class:`ADOPT` class.
"""

import math
from typing import Any, Callable, Optional, Union, cast

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _use_grad_for_differentiable,
    _view_as_real,
)
from torch.types import Number

from composer.optim.utils import get_report_curvature
from composer.utils import dist

__all__ = ['ADOPT', 'adopt']


def _default_clip_lambda(step: Number | Tensor | Any) -> float:
    r"""Return the default clipping value for the given step.

    By default, it uses :math:`\text{clip} = \text{step}^{0.25}`.

    Args:
        step (Number or Tensor or Any): The current iteration/step value
            (can be a scalar Python number or a PyTorch Tensor).

    Returns:
        float: The clipping threshold for the given step.
    """
    internal_step: int
    if isinstance(step, (Tensor)):
        internal_step = int(step.item())
    elif isinstance(step, (Number)):
        internal_step = int(step)
    else:
        raise TypeError(f'step must be a Number or Tensor, but got {type(step)}')
    return internal_step**0.25


class ADOPT(Optimizer):
    r"""Implements the ADOPT algorithm.

    This is a variant of Adam designed to
    converge optimally for any :math:`\beta_2`,
    featuring an adaptive clipping
    mechanism.

    :class:`ADOPT` builds upon Adam with an additional per-step clipping of
    normalized gradients. The clipping threshold can be controlled via a user-
    defined function ``clip_lambda(step)`` or the default
    :func:`_default_clip_lambda`.

    The default threshold is :math:`\text{step}^{0.25}`, which was found to
    stabilize updates over the training process.

    This optimizer shares many of Adam's hyperparameters but modifies
    the second moment computations and update rules to incorporate this
    gradient clipping logic.

    Args:
        params (iterable or ParamsT): Iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float or Tensor, optional): Learning rate. Default: 1e-3.
            If a ``Tensor`` is passed, ensure you set ``capturable=True``
            and ``foreach=False`` if you want to capture it in a graph.
        betas (Tuple[float, float], optional): Coefficients used for
            computing running averages of gradient and its square
            (default: (0.9, 0.9999)).
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-6).
        clip_lambda (callable, optional): A function that, given the current
            step (as a scalar), returns the clipping threshold. If
            ``None``, the default clipping rule
            :func:`_default_clip_lambda` is used. Default: None.
        weight_decay (float, optional): Weight decay coefficient (default: 0.0).
            If ``decouple=True``, it uses a decoupled weight decay
            (sometimes known as AdamW-style). Otherwise, it follows the
            original L2 regularization approach.
        decouple (bool, optional): Whether to decouple weight decay (like AdamW)
            or not. Default: False.
        foreach (bool, optional): Whether to use the multi-tensor APIs.
            When None, it will automatically decide. Default: None.
        maximize (bool, optional): Wheter to maximize rather than minimize
            the loss. Default: False.
        capturable (bool, optional): Whether the optimizer should allow
            gradient capture (for example, with CUDA graphs). Default: False.
        differentiable (bool, optional): Whether the optimizer is being used
            in a context that requires higher-order differentiation.
            Default: False.
        fused (bool, optional): Whether to use a fused version of the kernel
            (if supported by PyTorch and if no constraints like
            differentiability exist). Default: None.
        report_curvature: bool = False, Whether to report curvature metrics
            for each parameter. Default: False.

    Example:
        >>> import torch
        >>> from adopt_optimizer import ADOPT
        >>>
        >>> model = MyModel()
        >>> optimizer = ADOPT(model.parameters(), lr=1e-3)
        >>>
        >>> for input, target in data:
        ...     def closure():
        ...         optimizer.zero_grad()
        ...         loss = loss_fn(model(input), target)
        ...         loss.backward()
        ...         return loss
        ...     loss = optimizer.step(closure)

    .. note::
        This optimizer does not support sparse gradients.

    .. note::
        The default clipping threshold scales as :math:`\text{step}^{0.25}`,
        but you can override this by passing a custom ``clip_lambda(step)``
        function.

    .. warning::
        Passing a ``Tensor`` for ``lr`` is only supported if
        ``foreach=False`` or ``capturable=True``. Attempting to combine
        ``lr`` as a ``Tensor`` with ``foreach=True`` and
        ``capturable=False`` will raise an error.
    """

    metric_functions = {
        'l2_norm/moment': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/moment2': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg_sq']),
        'min/moment2': lambda param, optim_state, step_tensor: torch.min(optim_state['exp_avg_sq']),
        'max/moment2': lambda param, optim_state, step_tensor: torch.max(optim_state['exp_avg_sq']),
        'l2_norm/param': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/update': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
    }

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[Number | Tensor | Any], float]] = _default_clip_lambda,
        weight_decay: float = 0.0,
        decouple: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        report_curvature: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError('lr as a Tensor is not supported for capturable=False and foreach=True',)
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self.clip_lambda = clip_lambda

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group['initial_lr'] = group['lr']

        if fused:
            if differentiable:
                raise RuntimeError('`fused` does not support `differentiable`')
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError('`fused` and `foreach` cannot be `True` together.')
            # TODO: support fused
            raise RuntimeError('`fused` is not currently supported')\

        # NOTE: Added to avoid expensive metrics
        # calculations
        self.curvature_metric_function: Callable[[Tensor, str], dict[str, Tensor]] | None = None
        if report_curvature:
            self.curvature_metric_function = get_report_curvature()

    def __setstate__(self, state):
        """Set the state of the optimizer for backward compatibility.

        Specifically, this handles the presence or absence of certain keys
        in older checkpoints, ensures new keys are properly initialized,
        and converts 'step' values to Tensor if needed.

        Args:
            state (dict): The optimizer state to set.

        Returns:
            None
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('decouple', False)
            fused = group.setdefault('fused', None)
            for p in group['params']:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state['step']):  # type: ignore[reportGeneralTypeIssues]
                    step_val = float(p_state['step'])  # type: ignore[reportGeneralTypeIssues]
                    p_state['step'] = (  # type: ignore[reportGeneralTypeIssues]
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group['capturable'] or group['fused'] else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        """Initialize a parameter group for ADOPT updates.

        This internal helper function:
        - Filters out parameters that do not have gradients.
        - Checks for sparse gradients (raising an error if found).
        - Creates state entries (`step`, `exp_avg`, `exp_avg_sq`) if not already present.
        - Appends valid parameters and their states to the respective lists.

        Args:
            group (dict): A dictionary representing a parameter group and its settings.
            params_with_grad (list[Tensor]): List to store parameters that have gradients.
            grads (list[Tensor]): List to store gradients of those parameters.
            exp_avgs (list[Tensor]): List to store exponential moving averages of gradients.
            exp_avg_sqs (list[Tensor]): List to store exponential moving averages of squared gradients.
            state_steps (list[Tensor]): List to store the optimizer step counts for these parameters.

        Returns:
            bool: A flag indicating whether complex parameters exist in this group.
        """
        has_complex = False
        for p in group['params']:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError('ADOPT does not support sparse gradients')
            grads.append(p.grad)

            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                if group['fused']:
                    _device_dtype_check_for_fused(p)
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state['step'] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group['fused']),
                        device=p.device,
                    ) if group['capturable'] or group['fused'] else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p,
                    memory_format=torch.preserve_format,
                )
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(
                    p,
                    memory_format=torch.preserve_format,
                )
                # NOTE: We don't have anything related to AMSGrad here

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])

            # NOTE: We don't have anything related to AMSGrad here

            if group['differentiable'] and state['step'].requires_grad:
                raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode',)

            # Foreach without capturable does not support a tensor lr
            if (group['foreach'] and isinstance(group['lr'], Tensor) and not group['capturable']):
                raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True',)

            state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single ADOPT optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. This is useful for, e.g., line search or
                other techniques that re-evaluate the model multiple times per
                iteration.

        Returns:
            Any or None: The loss, if the closure is provided and called,
            otherwise None.

        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = cast(tuple[float, float], group['betas'])
            # NOTE: We don't have anything related to AMSGrad here

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                initial_lr=group['initial_lr'],
                decouple=group['decouple'],
                clip_lambda=self.clip_lambda,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, 'grad_scale', None),
                found_inf=getattr(self, 'found_inf', None),
                has_complex=has_complex,
            )

        return loss

    def dist_reduce_metrics(self, optimizer_metrics):
        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = dist.all_gather_object(local_keys)
        all_keys = set()
        for keys in all_gathered_keys:
            all_keys.update(keys)

        # Sort keys to ensure every rank has the same keys order
        # Only L2 norm metric keys are present, can apply regular sort
        all_keys = sorted(all_keys)
        for metric in all_keys:
            if metric.startswith('l2_norm'):
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')

                optimizer_metrics[metric] = math.sqrt(reduced)
            else:
                reduced = optimizer_metrics.get(metric, torch.tensor(0.0, device=torch.cuda.current_device()))
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation='SUM')
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics):
        """Preprocess metrics to reduce across ranks correctly."""
        # Only L2 norm metric keys are present, can skip sorting at this stage
        for metric in optimizer_metrics:
            # L2 norms need to be squared, before they are reduced via summation
            optimizer_metrics[metric] = optimizer_metrics[metric]**2

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']
        decouple = self.param_groups[0]['decouple']

        if param in self.state:
            param_optim_state = self.state[param]
            # NOTE: This is inverting the ADOPT update to recover the step tensor
            step_tensor = lr * param_optim_state['exp_avg']
            if weight_decay != 0 and decouple:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                scaling_factor = (decay_factor * weight_decay) / (1 - decay_factor * weight_decay)
                step_tensor.mul_(1 + scaling_factor).add_(param, alpha=scaling_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )
            # NOTE: these are heavy and require extra memory
            if self.curvature_metric_function is not None:
                optimizer_metrics.update(self.curvature_metric_function(param, name))

        return optimizer_metrics


ADOPT.__doc__ = (
    r"""Implements ADOPT algorithm.

    .. math::
    \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \alpha_t \text{ (lr)}, \: \beta_1, \beta_2 \text{ (betas)},
           \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)},
           \: \epsilon \text{ (epsilon)}, \: c_t \text{ (clip)}                              \\
        &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)},
           \quad v_0 \leftarrow g_0^2 \text{ (second moment)}                                 \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                          \\

        &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                        \\
        &\hspace{10mm}g_t \leftarrow -\nabla_{\theta} f(\theta_{t-1})                         \\
        &\hspace{5mm}\textbf{else}                                                            \\
        &\hspace{10mm}g_t \leftarrow \nabla_{\theta} f(\theta_{t-1})                          \\

        &\hspace{5mm}m_t \leftarrow \beta_1 \, m_{t-1} \;+\;
            \bigl(1 - \beta_1\bigr)\,\mathrm{Clip}\!\Bigl(
                \frac{g_t}{\max\{\sqrt{v_{t-1}}, \,\epsilon\}},\, c_t
            \Bigr)                                                                            \\

        &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} \;-\; \alpha_t \, m_t                  \\
        &\hspace{5mm}v_t \leftarrow \beta_2 \, v_{t-1} \;+\;
            \bigl(1 - \beta_2\bigr)\,g_t^2                                                    \\
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        &\bf{return} \:  \theta_t                                                     \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
   \end{aligned}

    For further details regarding the algorithm we refer to the original research paper_.
    """ + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_maximize_doc}
        {_foreach_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    .. _ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate:
        https://arxiv.org/abs/2411.02853
    """
)


def _single_tensor_adopt(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    initial_lr: Optional[float],
    decouple: bool,
    clip_lambda: Optional[Callable[[Number | Tensor | Any], float]],
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    """Single-tensor implementation of ADOPT step.

    Args:
        params (list[Tensor]): List of parameters to be updated.
        grads (list[Tensor]): List of gradients corresponding to the parameters.
        exp_avgs (list[Tensor]): List of first moment (EMA of grads).
        exp_avg_sqs (list[Tensor]): List of second moment (EMA of squared grads).
        state_steps (list[Tensor]): List of step counters for each parameter.
        grad_scale (Tensor or None): Not used in this implementation.
        found_inf (Tensor or None): Not used in this implementation.
        initial_lr (float or None): The initial learning rate to compute
            the ratio for decoupled weight decay.
        decouple (bool): Whether to apply decoupled weight decay (AdamW style).
        clip_lambda (Callable or None): Function returning the clipping threshold
            given the step.
        beta1 (float): Exponential decay rate for the first moment.
        beta2 (float): Exponential decay rate for the second moment.
        lr (float or Tensor): Learning rate.
        weight_decay (float): Weight decay coefficient.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Whether to maximize rather than minimize.
        capturable (bool): Whether this optimizer should allow graph capture.
        differentiable (bool): Whether this optimizer is used in a context
            requiring differentiable optimization.
        has_complex (bool): Indicates if any parameters are complex.

    Raises:
        RuntimeError: If any constraints (e.g., sparse gradients) are violated.
    """
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:  # type: ignore[reportGeneralTypeIssues]
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type and param.device.type in capturable_supported_devices
            ), f'If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}.'

        step = step_t if capturable or differentiable else _get_value(step_t)

        # Perform stepweight decay if not decoupled
        if weight_decay != 0 and not decouple:
            # NOTE: The decay factor follows the same schedule of the learning rate w/o being scaled by it
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # NOTE: During the zero-th step (when `step == 0``), the algorithm doesn't update the weights but just initializes the second momenta vectors
        if step == 0:
            # NOTE: Compared to the original implementation, we removed a useless `.conj()` call to the second `grad` in the function below
            exp_avg_sq.addcmul_(grad, grad)
            # update step
            step_t += 1
            # Here, we skip whatever happens below moving to the next iteration of the for loop
            continue

        # Perform stepweight decay if decoupled
        if weight_decay != 0 and decouple:
            # NOTE: The decay factor follows the same schedule of the learning rate w/o being scaled by it
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            param.mul_(1 - decay_factor * weight_decay)

        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)
        if clip_lambda is not None:
            clip = clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        exp_avg.lerp_(normed_grad, 1 - beta1)

        param.add_(exp_avg, alpha=-_get_value(lr))
        # NOTE: Compared to the original implementation, we removed a useless `.conj()` call to the second `grad` in the function below
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # update step
        step_t += 1


def _multi_tensor_adopt(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    initial_lr: Optional[float],
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[Number | Tensor | Any], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    """Multi-tensor implementation of the ADOPT step.

    This performs updates on groups of tensors to improve
    performance when many parameters are involved.

    Args:
        params (list[Tensor]): List of parameters to be updated.
        grads (list[Tensor]): List of gradients corresponding to the parameters.
        exp_avgs (list[Tensor]): List of first moment (EMA of grads).
        exp_avg_sqs (list[Tensor]): List of second moment (EMA of squared grads).
        state_steps (list[Tensor]): List of step counters for each parameter.
        grad_scale (Tensor or None): Not used in this implementation.
        found_inf (Tensor or None): Not used in this implementation.
        initial_lr (float or None): The initial learning rate to compute
            the ratio for decoupled weight decay.
        has_complex (bool): Indicates if any parameters are complex.
        beta1 (float): Exponential decay rate for the first moment.
        beta2 (float): Exponential decay rate for the second moment.
        lr (float or Tensor): Learning rate.
        clip_lambda (Callable or None): Function returning the clipping threshold
            given the step.
        weight_decay (float): Weight decay coefficient.
        decouple (bool): Whether to apply decoupled weight decay (AdamW style).
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Whether to maximize rather than minimize.
        capturable (bool): Whether this optimizer should allow graph capture.
        differentiable (bool): Whether this optimizer is used in a context
            requiring differentiable optimization.

    Raises:
        RuntimeError: If ``lr`` is a ``Tensor`` combined with
            ``capturable=False`` and ``foreach=True``, or if the JIT compiler
            usage is not supported.
    """
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True',)

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:  # type: ignore[reportGeneralTypeIssues]
        capturable_supported_devices = _get_capturable_supported_devices(supports_xla=False,)
        assert all(
            p.device.type == step.device.type and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f'If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}.'

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps],  # type: ignore[reportArgumentType]
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            _view_as_real(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
            )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0 and not decouple:
            # NOTE: The decay factor follows the same schedule of the learning rate w/o being scaled by it
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            weight_decay_unscaled = decay_factor * weight_decay
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=_get_value(weight_decay_unscaled))
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=-_get_value(weight_decay_unscaled),
                )

        # NOTE: During the zero-th step (when `device_state_steps[0] == 0``), the algorithm doesn't update the weights but just initializes the second momenta vectors
        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)

            # Update steps
            # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
            # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
            # wrapped it once now. The alpha is required to assure we go to the right overload.
            if (
                not torch._utils.is_compiling()  # type: ignore[reportGeneralTypeIssues]
                and device_state_steps[0].is_cpu
            ):
                torch._foreach_add_(
                    device_state_steps,
                    torch.tensor(1.0, device='cpu'),
                    alpha=1.0,
                )
            else:
                torch._foreach_add_(device_state_steps, 1)
            # Here, we skip whatever happens below moving to the next iteration of the for loop
            continue

        if weight_decay != 0 and decouple:
            # NOTE: The decay factor follows the same schedule of the learning rate w/o being scaled by it
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            weight_decay_unscaled = decay_factor * weight_decay
            torch._foreach_add_(device_params, device_params, alpha=-_get_value(weight_decay_unscaled))

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)
        if clip_lambda is not None:
            clip = clip_lambda(_get_value(device_state_steps[0]))
            torch._foreach_maximum_(normed_grad, -clip)
            torch._foreach_minimum_(normed_grad, clip)

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-_get_value(lr))
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs,
            device_grads,
            device_grads,
            value=1 - beta2,
        )

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:  # type: ignore[reportGeneralTypeIssues]
            torch._foreach_add_(
                device_state_steps,
                torch.tensor(1.0, device='cpu'),
                alpha=1.0,
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


def _fused_adopt(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    initial_lr: Optional[float],
    has_complex: bool,  # Needed for consistency.
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[int], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
) -> None:
    """Fused implementation of the ADOPT step.

    .. warning::
        This is not yet implemented. It raises :class:`NotImplementedError`.

    Args:
        params (list[Tensor]): List of parameters.
        grads (list[Tensor]): List of gradients.
        exp_avgs (list[Tensor]): List of first moment (EMA of grads).
        exp_avg_sqs (list[Tensor]): List of second moment (EMA of squared grads).
        state_steps (list[Tensor]): List of step counters.
        grad_scale (Tensor or None): Gradient scaling factor (if any).
        found_inf (Tensor or None): Whether infinite grads were found (amp).
        initial_lr (float or None): The initial learning rate for decoupled weight decay.
        has_complex (bool): Indicates if any parameters are complex.
        beta1 (float): Exponential decay rate for the first moment.
        beta2 (float): Exponential decay rate for the second moment.
        lr (float or Tensor): Learning rate.
        clip_lambda (Callable or None): Function returning the clipping threshold.
        weight_decay (float): Weight decay coefficient.
        decouple (bool): Whether to apply decoupled weight decay.
        eps (float): Epsilon for numerical stability.
        maximize (bool): Whether to maximize rather than minimize.
        capturable (bool): Whether this optimizer should allow graph capture.
        differentiable (bool): Whether differentiable optimization is needed.

    Raises:
        NotImplementedError: This function is not implemented.
    """
    raise NotImplementedError


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)
def adopt(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    initial_lr: Optional[float] = None,
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    clip_lambda: Optional[Callable[[Number | Tensor | Any], float]],
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs the ADOPT algorithm step.

    It directly applies the parameter updates in-place, modifying:
    - ``params`` in place.
    - ``exp_avgs`` (first moment).
    - ``exp_avg_sqs`` (second moment).
    - ``state_steps`` (step counters).

    It provides options to use different internal implementations:
    - Single-tensor (default).
    - Multi-tensor (``foreach=True``).
    - Fused (``fused=True``, if supported).

    The fused implementation is currently unimplemented and raises an error.

    Args:
        params (list[Tensor]): List of parameters to update.
        grads (list[Tensor]): List of gradients of the same shape as ``params``.
        exp_avgs (list[Tensor]): Exponential moving averages of the gradients.
        exp_avg_sqs (list[Tensor]): Exponential moving averages of the
            squared gradients.
        state_steps (list[Tensor]): List of step counters.
        initial_lr (float or None, optional): The initial learning rate used
            to compute the ratio for decoupled weight decay. If None,
            weight decay will not be scaled by any ratio. Default: None.
        foreach (bool, optional): Whether to use multi-tensor APIs.
            Defaults to None, which auto-selects.
        capturable (bool, optional): Whether this optimizer should allow
            graph capture. Default: False.
        differentiable (bool, optional): Whether this function is used
            in a context requiring differentiable optimization.
            Default: False.
        fused (bool, optional): Whether to use a fused version, if available.
            Currently not implemented and will raise an error if True.
            Default: None.
        grad_scale (Tensor or None): Not used in the current implementation,
            typically relevant for mixed-precision training.
        found_inf (Tensor or None): Not used in the current implementation,
            typically relevant for mixed-precision training.
        has_complex (bool, optional): Indicates whether any of the parameters
            are complex. Default: False.

        beta1 (float): Coefficient for computing running averages of gradient.
        beta2 (float): Coefficient for computing running averages of
            squared gradient.
        lr (float or Tensor): Learning rate.
        clip_lambda (Callable or None): A function taking the current step
            as input and returning a gradient clipping threshold. Defaults
            to :func:`_default_clip_lambda` if None.
        weight_decay (float): Weight decay coefficient.
        decouple (bool): Whether to apply decoupled weight decay.
        eps (float): Term added to the denominator for numerical stability.
        maximize (bool): Whether to maximize rather than minimize.

    Raises:
        RuntimeError: If the fused implementation is requested (``fused=True``)
            but is not supported yet, or if incompatible combinations of
            ``lr``, ``capturable``, and ``foreach`` are passed.
        AssertionError: If the environment does not support certain capture
            options or if the user attempts to use multi-tensor/differentiable
            combos that are disallowed.

    Example:
        >>> import torch
        >>> from adopt_optimizer import adopt, _default_clip_lambda
        >>>
        >>> # Suppose p, grad, exp_avg, exp_avg_sq, step are Tensors
        >>> p = torch.randn(3, requires_grad=True)
        >>> grad = torch.randn_like(p)
        >>> exp_avg = torch.zeros_like(p)
        >>> exp_avg_sq = torch.zeros_like(p)
        >>> step = torch.zeros((), dtype=torch.float)
        >>>
        >>> adopt(
        ...     params=[p], grads=[grad],
        ...     exp_avgs=[exp_avg], exp_avg_sqs=[exp_avg_sq],
        ...     state_steps=[step],
        ...     lr=0.001,
        ...     beta1=0.9, beta2=0.9999,
        ...     eps=1e-6,
        ...     weight_decay=0.0,
        ...     decouple=False,
        ...     clip_lambda=_default_clip_lambda,  # optional, can be None
        ...     maximize=False,
        ... )
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params,
            differentiable,
            use_fused=False,
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if fused and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with fused optimizers')

    if fused and not torch.jit.is_scripting():
        func = _fused_adopt
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        initial_lr=initial_lr,
        decouple=decouple,
        clip_lambda=clip_lambda,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
