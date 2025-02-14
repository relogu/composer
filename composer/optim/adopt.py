# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the ADOPT optimizer.

Paper: https://arxiv.org/abs/2411.02853
Original code: https://github.com/iShohei220/adopt

"""

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

__all__ = ['ADOPT', 'adopt']


def _default_clip_lambda(step: Number | Tensor | Any) -> float:
    """Return the default clipping value for the given step."""
    internal_step: int
    if isinstance(step, (Tensor)):
        internal_step = int(step.item())
    elif isinstance(step, (Number)):
        internal_step = int(step)
    else:
        raise TypeError(f'step must be a Number or Tensor, but got {type(step)}')
    return internal_step**0.25


# TODO: Add docstrings
class ADOPT(Optimizer):
    """ADOPT optimizer."""

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
            raise RuntimeError('`fused` is not currently supported')

    def __setstate__(self, state):
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
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
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


# TODO: Fix it
ADOPT.__doc__ = (
    r"""Implements ADOPT algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
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
    r"""Functional API that performs ADOPT algorithm computation.

    TODO: Add more details
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
