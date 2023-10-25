import pdb

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch_geometric.utils import scatter

from typing import TypedDict, Iterator, IO, Any

from .model import ModelCfg, Model
from .batching import Batch
from .utils.modules import binary_cross_entropy_with_logits


class TrainCfg(TypedDict):
    model_config:       ModelCfg
    num_epochs:         int
    warmup_epochs:      int
    warmdown_epochs:    int
    batch_size_s:       int
    batch_size_h:       int
    max_scope_size:     int
    max_ast_len:        int
    backprop_every:     int
    max_lr:             float
    min_lr:             float
    max_scope_size:     int
    max_ast_len:        int
    train_files:        list[str]
    dev_files:          list[str]
    test_files:         list[str]


class TrainStats(TypedDict):
    loss:               list[float]
    predictions:        list[bool]
    truths:             list[bool]


def _add(x: TrainStats, y: TrainStats) -> TrainStats:
    return {'loss': x['loss'] + y['loss'],
            'predictions': x['predictions'] + y['predictions'],
            'truths': x['truths'] + y['truths']}


def binary_stats(predictions: list[bool], truths: list[bool]) -> tuple[int, int, int, int]:
    tp = sum((x for x, y in zip(predictions, truths) if y))
    fn = sum((not x for x, y in zip(predictions, truths) if y))
    tn = sum((not x for x, y in zip(predictions, truths) if not y))
    fp = sum((x for x, y in zip(predictions, truths) if not y))
    return tp, fn, tn, fp


def _macro_binary_stats(tp: int, fn: int, tn: int, fp: int) -> tuple[float, float, float, float]:
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, f1, prec, rec


def macro_binary_stats(predictions: list[bool], truths: list[bool]) -> tuple[float, float, float, float]:
    return _macro_binary_stats(*binary_stats(predictions, truths))


def acc(xs: list[bool]) -> float:
    return sum(xs)/len(xs)


def subsample_mask(xs: Tensor, factor: float) -> Tensor:
    num_neg_samples = min(xs.sum() * factor, (~xs).sum())
    false_indices = torch.nonzero(~xs)
    sampled_false_indices = false_indices[torch.randperm(false_indices.size(0))][:num_neg_samples]
    mask = torch.zeros_like(xs)
    mask[xs] = True
    mask[sampled_false_indices] = True
    return mask


class Trainer(Model):
    def compute_loss(self, batch: Batch) -> tuple[list[bool], list[bool], Tensor]:
        scope_reprs, hole_reprs = self.encode(batch)
        predictions = self.predict_lemmas(scope_reprs=scope_reprs,
                                          hole_reprs=hole_reprs,
                                          edge_index=batch.edge_index)
        loss = binary_cross_entropy_with_logits(
            input=predictions,
            target=batch.premises.float(),
            reduction='none')
        loss = scatter(loss, batch.edge_index[1], reduce='sum')
        return (predictions.sigmoid().round().cpu().bool().tolist(),
                batch.premises.cpu().tolist(),
                loss)

    def train_epoch(self,
                    epoch: Iterator[Batch],
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop_every: int) -> TrainStats:
        self.train()

        epoch_stats: TrainStats = {'loss': [], 'predictions': [], 'truths': []}
        for i, batch in enumerate(epoch):
            batch_stats = self.train_batch(
                batch=batch, optimizer=optimizer, scheduler=scheduler, backprop=(i + 1) % backprop_every == 0)
            epoch_stats = _add(epoch_stats, batch_stats)
        return epoch_stats

    def train_batch(self,
                    batch: Batch,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop: bool) -> TrainStats:
        predictions, truths, loss = self.compute_loss(batch)
        loss.mean().backward()

        if backprop:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return {'loss': loss.tolist(), 'predictions': predictions, 'truths': truths}

    def eval_batch(self, batch: Batch) -> TrainStats:
        predictions, truths, loss = self.compute_loss(batch)
        return {'loss': loss.tolist(), 'predictions': predictions, 'truths': truths}

    def eval_epoch(self, epoch: Iterator[Batch]) -> TrainStats:
        self.eval()
        epoch_stats: TrainStats = {'loss': [], 'predictions': [], 'truths': []}

        with torch.no_grad():
            for i, batch in enumerate(epoch):
                epoch_stats = _add(epoch_stats, self.eval_batch(batch))
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