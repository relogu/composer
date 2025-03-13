# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor vs during training."""

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['QuasiHyperbolicMonitor']


class QuasiHyperbolicMonitor(Callback):
    """Logs hyperbolic v1 parameter.

    This callback iterates over all optimizers and their parameter groups to log v1 under the ``v1-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` key.

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

    The vs are logged by the :class:`.Logger` to the following key as described
    below.

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    |                                             | vs for each optimizer and  |
    | ``v_{idx}-{OPTIMIZER_NAME}/group{GROUP_NUMBER}`` | parameter group for that optimizer is |
    |                                             | logged to a separate key.             |
    +---------------------------------------------+---------------------------------------+
    """

    def __init__(self) -> None:
        pass

    def batch_end(self, state: State, logger: Logger):
        assert state.optimizers is not None, 'optimizers must be defined'
        step = state.timestamp.batch.value
        for optimizer in state.optimizers:
            vs: list[float] = [group['v'] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for idx, v1 in enumerate(vs):
                logger.log_metrics({f'v1-{name}/group{idx}': v1}, step)
