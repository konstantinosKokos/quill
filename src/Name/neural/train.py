from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import TypedDict, Iterator

from .model import ModelCfg, Model
from .batching import Batch


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
    train_files:        list[str]
    dev_files:          list[str]
    test_files:         list[str]


class TrainStats(TypedDict):
    loss:               float
    predictions:        list[bool]
    truth:              list[bool]


def binary_stats(predictions: list[bool], truths: list[bool]) -> tuple[int, int, int, int]:
    tp = sum([x == y for x, y in zip(predictions, truths) if y])
    fn = sum([x != y for x, y in zip(predictions, truths) if y])
    tn = sum([x == y for x, y in zip(predictions, truths) if not y])
    fp = sum([x != y for x, y in zip(predictions, truths) if not y])
    return tp, fn, tn, fp


def macro_binary_stats(tp: int, fn: int, tn: int, fp: int) -> tuple[float, float, float, float]:
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, f1, prec, rec


class Trainer(Model):
    def train_epoch(self,
                    epoch: Iterator[Batch],
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop_every: int) -> TrainStats:
        epoch_stats = {'loss': 0, 'predictions': [], 'truth': []}
        for i, batch in enumerate(epoch):
            batch_stats = self.train_batch(
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                backprop=(i + 1) % backprop_every == 0)
            epoch_stats['loss'] += batch_stats['loss']
            epoch_stats['predictions'] += batch_stats['predictions']
            epoch_stats['truth'] += batch_stats['truth']
        return epoch_stats

    def train_batch(self,
                    batch: Batch,
                    optimizer: Optimizer,
                    scheduler: LRScheduler,
                    backprop: bool) -> TrainStats:
        predictions, truth, loss = self.compute_loss(batch)
        loss.backward()

        if backprop:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return {'loss': loss.item(),
                'predictions': predictions,
                'truth': truth}
