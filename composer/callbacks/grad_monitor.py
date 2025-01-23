# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during DDP training."""

from typing import Optional
import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ["GradMonitor"]

class GradMonitor(Callback):
    """extracts gradients from the self.state.model during training.

    The extracted gradients are stored in the self.self.grads attribute, in the form of a list of tensors.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import GradMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[GradMonitor()],
            ... )

    """

    def __init__(
        self,
        device: str,
    ) -> None:
        self.executed_steps = 0
        # NOTE: May want to make this configurable
        self.device = torch.device(device)
        self.grads: Optional[dict[str, torch.Tensor]] = None

    def _extract_grads(
        self, state: State
    ) -> None:
        """Extracts gradients of each batch from the model
        A running average of the gradients is stored in the state.

        Args:
            state (State): The state object.
            device (torch.device, optional): The device to store the gradients. Defaults to CPU.
        """

        group = list(state.model.named_parameters())
        grad_dict: dict[str, torch.Tensor] = {}
        for name, p in group:
            if p.grad is not None and p.requires_grad:
                grad_dict[name] = p.grad.to(self.device).detach().clone()

        # average the gradients
        prev_grads = self.grads
        if prev_grads:
            aver_grad_dict = {
                name: (
                    (prev_grads[name] * self.executed_steps + grad_dict[name])
                    / (self.executed_steps + 1)
                )
                for name in prev_grads
            }
        else:  # the first batch, no need to average
            aver_grad_dict = grad_dict
        self.executed_steps = self.executed_steps + 1

        self.grads = aver_grad_dict

    def after_backward(self, state: State, logger: Logger) -> None:
        """Extract gradients on event ``Event.AFTER_BACKWARD`` in the function of _train_microbatch."""
        assert hasattr(state, "is_final_microbatch"), "Attribute required to avoid inefficiency"
        if state.is_final_microbatch: # type: ignore[reportAttributeAccessIssue]
            self._extract_grads(state)
        
    def batch_end(self, state: State, logger: Logger) -> None:
        """Sync gradient store on ``Event.BATCH_END`` in the function of _train_microbatch."""
        assert self.grads is not None, "self.grads should not be None if this callback is used"
        if dist.is_initialized() and dist.get_world_size() > 1:
            for name in self.grads.keys():
                last_grad = self.grads[name]
                
                dist.all_reduce(last_grad)
                last_grad.div_(dist.get_world_size())
                
                # Should not be necessary, but just in case
                self.grads[name] = last_grad
        