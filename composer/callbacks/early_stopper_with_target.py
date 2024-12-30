from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['EarlyStopperWithTarget']


class EarlyStopperWithTarget(Callback):

    def __init__(
        self,
        monitor: str,
        dataloader_label: str,
        target_value: float,
        comp: Optional[Union[str, Callable[[Any, Any], Any]]] = None,
        min_delta: float = 0.0,
        patience: Union[int, str, Time] = 1,
    ):
        self.monitor = monitor
        self.dataloader_label = dataloader_label
        self.target_value = target_value
        self.min_delta = abs(min_delta)
        if callable(comp):
            self.comp_func = comp
        if isinstance(comp, str):
            if comp.lower() in ('greater', 'gt'):
                self.comp_func = torch.greater
            elif comp.lower() in ('less', 'lt'):
                self.comp_func = torch.less
            else:
                raise ValueError(
                    "Unrecognized comp string. Use the strings 'gt', 'greater', 'lt' or 'less' or a callable comparison operator",
                )
        if comp is None:
            if any(substr in monitor.lower() for substr in ['loss', 'error', 'perplexity']):
                self.comp_func = torch.less
            else:
                self.comp_func = torch.greater

        if isinstance(patience, str):
            self.patience = Time.from_timestring(patience)
        elif isinstance(patience, int):
            self.patience = Time(patience, TimeUnit.EPOCH)
        else:
            self.patience = patience
            if self.patience.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
                raise ValueError('If `patience` is an instance of Time, it must have units of EPOCH or BATCH.')

    def _get_monitored_metric(self, state: State):
        if self.dataloader_label == 'train' and state.train_metrics is not None:
            if self.monitor in state.train_metrics:
                return state.train_metrics[self.monitor].compute()
        else:
            if self.monitor in state.eval_metrics[self.dataloader_label]:
                return state.eval_metrics[self.dataloader_label][self.monitor].compute()
        raise ValueError(
            f"Couldn't find the metric {self.monitor} with the dataloader label {self.dataloader_label}."
            "Check that the dataloader_label is set to 'eval', 'train' or the evaluator name.",
        )

    def _update_stopper_state(self, state: State):
        # Check if we passed the patience threshold
        is_patience_exceeded = False
        if self.patience.unit == TimeUnit.EPOCH:
            if state.timestamp.epoch > self.patience:
                is_patience_exceeded = True
        elif self.patience.unit == TimeUnit.BATCH:
            if state.timestamp.batch > self.patience:
                is_patience_exceeded = True
        else:
            raise ValueError(f'The units of `patience` should be EPOCH or BATCH.')
        
        # If we passed the patience threshold, check the value of the monitored metric
        if is_patience_exceeded:
            # Get the monitored metric
            metric_val = self._get_monitored_metric(state)
            # Compare the monitored metric to the target value
            if self.comp_func(metric_val, self.target_value) and torch.abs(metric_val - self.target_value) > self.min_delta:
                state.stop_training()

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is an eval metric or in an evaluator
            self._update_stopper_state(state)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if self.dataloader_label == state.dataloader_label:
            # if the monitored metric is not an eval metric, the right logic is run on EPOCH_END
            self._update_stopper_state(state)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.patience.unit == TimeUnit.BATCH and self.dataloader_label == state.dataloader_label:
            self._update_stopper_state(state)
