import os
import pickle

from src.Name.neural.train import TrainCfg, Trainer, acc, Logger, ModelCfg, macro_binary_stats
from src.Name.neural.batching import filter_data, Sampler, Collator
from src.Name.neural.utils.schedules import make_schedule

from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from random import seed

import sys


def train(config: TrainCfg, data_path: str, cast_to: str):
    logger = Logger(sys.stdout, './log.txt')
    sys.stdout = logger

    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    files = list(filter_data(
        files=files,
        max_scope_size=config['max_scope_size'],
        max_ast_len=config['max_ast_len']))
    print(f'Kept {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    train_files = [file for file in files if file.file.name in config['train_files']]
    dev_files = [file for file in files if file.file.name in config['dev_files']]
    print(f'Training on {len(train_files)} files with {sum(len(file.hole_asts) for file in train_files)} holes.')
    print(f'Evaluating on {len(dev_files)} files with {sum(len(file.hole_asts) for file in train_files)} holes.')

    train_sampler = Sampler(train_files)
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, device=cast_to, allow_self_loops=False)

    model = Trainer(config['model_config']).to(device(cast_to))
    optimizer = AdamW(params=model.parameters(), lr=1, weight_decay=1e-02)
    schedule = make_schedule(warmup_steps=config['warmup_epochs'] * epoch_size,
                             warmdown_steps=config['warmdown_epochs'] * epoch_size,
                             max_lr=config['max_lr'],
                             min_lr=config['min_lr'],
                             total_steps=config['num_epochs'] * epoch_size)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=schedule, last_epoch=-1)

    best_loss = 1e10

    for epoch in range(config['num_epochs']):
        print(f'Epoch {epoch}')
        print('-' * 64)
        train_epoch = model.train_epoch(
            epoch=map(collator, train_sampler.iter(
                batch_size_s=config['batch_size_s'],
                batch_size_h=config['batch_size_h'])),
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_every=config['backprop_every'])
        print(f'Train loss: {sum(train_epoch["loss"])/len(train_epoch["predictions"])}')
        print(f'Train stats: {macro_binary_stats(train_epoch["predictions"], train_epoch["truths"])}')
        dev_epoch = model.eval_epoch(map(lambda x: collator([x]), dev_files))
        print(f'Dev loss: {sum(dev_epoch["loss"])/len(dev_epoch["predictions"])}')
        print(f'Dev stats: {macro_binary_stats(dev_epoch["predictions"], dev_epoch["truths"])}')
        print()

        # if sum(dev_epoch['loss']) < best_loss:
        #     print('Saving...')
        #     model.save(f'./model.pt')
        #     best_loss = sum(dev_epoch['loss'])
        # print('=' * 64 + '\n')


if __name__ == '__main__':
    seed(42)
    # todo.
    files = [os.path.splitext(file)[0] for file in os.listdir('../data/stdlib/')]
    # stdlib = [line for line in open('./data/stdlib.contents').read().split('\n')]
    # unimath = [line for line in open('./data/um.contents').read().split('\n')]
    # typetopo = [line for line in open('./data/tt.contents').read().split('\n')]
    # shuffle(stdlib)

    model_config: ModelCfg = {
        'depth': 8,
        'num_heads': 8,
        'dim': 128,
        'atn_dim': None,
        'dropout_rate': 0.15,
    }

    train_cfg: TrainCfg = {
        'model_config': model_config,
        'num_epochs': 99,
        'warmup_epochs': 3,
        'warmdown_epochs': 90,
        'batch_size_s': 1,
        'batch_size_h': 8,
        'max_lr': 5e-4,
        'min_lr': 1e-7,
        'backprop_every': 1,
        'train_files': [f for f in files if f != 'Simple'],
        'dev_files': [],
        'test_files': [],
        'max_scope_size': 300,
        'max_ast_len': 100,
    }

    train(train_cfg, '../data/tokenized.p', 'cuda')
