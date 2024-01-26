from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from typing import TypedDict, Iterator, IO, Any
from dataclasses import dataclass

from .model import ModelCfg, Model
from .batching import Batch
from .utils.ranking import infoNCE, evaluate_rankings


class TrainCfg(TypedDict):
    model_config:       ModelCfg
    num_epochs:         int
    warmup_epochs:      int
    warmdown_epochs:    int
    batch_size_s:       int
    batch_size_h:       int
    max_tokens:         int
    backprop_every:     int
    max_lr:             float
    min_lr:             float
    train_files:        list[str]
    dev_files:          list[str]
    test_files:         list[str]
    allow_self_loops:   bool
    half_precision:     bool


@dataclass
class TrainStats:
    loss:               tuple[float, ...] = ()
    ap:                 tuple[float, ...] = ()
    rp:                 tuple[float, ...] = ()

    def __add__(self, other: TrainStats) -> TrainStats:
        return TrainStats(loss=self.loss + other.loss, ap=self.ap + other.ap, rp=self.rp + other.rp)


class Trainer(Model):
    def compute_loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
        scope_reprs, hole_reprs = self.encode(batch)
        predictions = self.match(
                scope_reprs=scope_reprs,
                hole_reprs=hole_reprs,
                edge_index=batch.edge_index)
        return predictions, infoNCE(predictions, batch.premises.bool(), batch.edge_index)

    def to_stats(self, batch: Batch, predictions: Tensor, loss: Tensor) -> TrainStats:
        zipped = tuple(zip(*evaluate_rankings(predictions, batch.edge_index[1], batch.premises)))
        ap, rp = zipped if zipped else ((), ())
        return TrainStats(loss=tuple(loss.tolist()),
                          ap=ap,
                          rp=rp)

    def train_epoch(self,
                    epoch: Iterator[Batch],
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop_every: int) -> TrainStats:
        self.train()

        epoch_stats = TrainStats()
        for i, batch in enumerate(epoch):
            epoch_stats += self.train_batch(
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                backprop=(i + 1) % backprop_every == 0)
        return epoch_stats

    def train_batch(self,
                    batch: Batch,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop: bool) -> TrainStats:
        predictions, loss = self.compute_loss(batch)
        loss.mean().backward()

        if backprop:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return self.to_stats(batch, predictions, loss)

    def eval_batch(self, batch: Batch) -> TrainStats:
        predictions, loss = self.compute_loss(batch)
        return self.to_stats(batch, predictions, loss)

    def eval_epoch(self, epoch: Iterator[Batch]) -> TrainStats:
        self.eval()
        epoch_stats = TrainStats()

        with torch.no_grad():
            for i, batch in enumerate(epoch):
                epoch_stats += self.eval_batch(batch)
        return epoch_stats


class Logger:
    def __init__(self, stdout: IO[str], log: str):
        self.stdout = stdout
        self.log = log

    def write(self, obj: Any) -> None:
        with open(self.log, 'a') as f:
            f.write(f'{obj}')
        self.stdout.write(f'{obj}')

    def flush(self):
        self.stdout.flush()
