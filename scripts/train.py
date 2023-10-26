import os
import pickle

from Name.nn.training import TrainCfg, Trainer, Logger, ModelCfg
from Name.nn.batching import filter_data, Sampler, Collator
from Name.nn.utils.schedules import make_schedule

from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from random import seed, shuffle

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
    print(f'Evaluating on {len(dev_files)} files with {sum(len(file.hole_asts) for file in dev_files)} holes.')

    train_sampler = Sampler(train_files)
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, device=cast_to, allow_self_loops=config['allow_self_loops'])

    model = Trainer(config['model_config']).to(device(cast_to))
    optimizer = AdamW(params=model.parameters(), lr=1, weight_decay=1e-02)
    schedule = make_schedule(warmup_steps=config['warmup_epochs'] * epoch_size,
                             warmdown_steps=config['warmdown_epochs'] * epoch_size,
                             max_lr=config['max_lr'],
                             min_lr=config['min_lr'],
                             total_steps=config['num_epochs'] * epoch_size)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=schedule, last_epoch=-1)

    best_ap = -1e08

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
        print(f'Train loss: {sum(train_epoch.loss)/len(train_epoch.loss)}')
        print(f'Train mAP: {sum(train_epoch.ap)/len(train_epoch.ap)}')
        print(f'Train R-Precision: {sum(train_epoch.rp) / len(train_epoch.rp)}')
        dev_epoch = model.eval_epoch(map(lambda x: collator([x]), dev_files))
        print(f'Dev loss: {sum(dev_epoch.loss)/len(dev_epoch.loss)}')
        print(f'Dev mAP: {sum(dev_epoch.ap) / len(dev_epoch.ap)}')
        print(f'Dev R-Precision: {sum(dev_epoch.rp) / len(dev_epoch.rp)}')
        if sum(dev_epoch.ap) > best_ap:
            print('Saving...')
            model.save(f'./model.pt')
            best_ap = sum(dev_epoch.ap)
        print('=' * 64 + '\n')


model_cfg: ModelCfg = {
    'depth': 6,
    'num_heads': 8,
    'dim': 128,
    'atn_dim': None,
    'dropout_rate': 0.15,
}

seed(42)
files = [os.path.splitext(file)[0] for file in os.listdir('./data/stdlib/')]
shuffle(files)
train_files, dev_files = files[:(int(0.75 * len(files)))], files[int(0.75 * len(files)):]

train_cfg: TrainCfg = {
    'model_config': model_cfg,
    'num_epochs': 99,
    'warmup_epochs': 3,
    'warmdown_epochs': 90,
    'batch_size_s': 2,
    'batch_size_h': 8,
    'max_lr': 5e-4,
    'min_lr': 1e-7,
    'backprop_every': 1,
    'train_files': train_files,
    'dev_files': dev_files,
    'test_files': [],
    'max_scope_size': 300,
    'max_ast_len': 100,
    'allow_self_loops': False
}


if __name__ == '__main__':
    train(train_cfg, './data/tokenized.p', 'cuda')
