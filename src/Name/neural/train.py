from .model import ModelCfg, Model
from .batching import Batch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import TypedDict, Iterator


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
    predictions:        list[int]
    truth:              list[int]


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
