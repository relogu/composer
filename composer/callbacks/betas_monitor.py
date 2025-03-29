# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor betas during training."""
from typing import cast

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['BetasMonitor']


class BetasMonitor(Callback):
    """Logs the betas.

    This callback iterates over all optimizers and their parameter groups to log
    betas under the ``beta_1-{OPTIMIZER_NAME}/group{GROUP_NUMBER}``, ``beta_2-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` keys.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import LRMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[LRMonitor()],
            ... )

    The betas are logged by the :class:`.Logger` to the following key as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    |                                             | betas for each optimizer and  |
    | ``beta_{idx}-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` | parameter group for that optimizer is |
    |                                             | logged to a separate key.             |
    +---------------------------------------------+---------------------------------------+
    """

    def __init__(self) -> None:
        pass

    def batch_end(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        step = state.timestamp.batch.value
        for optimizer in state.optimizers:
            betas = [group['betas'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for idx, beta_tuple in enumerate(betas):
                beta1, beta2 = cast(tuple[float, float], beta_tuple)
                logger.log_metrics({f'beta1-{name}/group{idx}': beta1}, step)
                logger.log_metrics({f'beta2-{name}/group{idx}': beta2}, step)
