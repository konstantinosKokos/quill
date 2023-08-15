import pickle

from src.Name.neural.train import TrainCfg, Trainer, acc, Logger, ModelCfg
from src.Name.neural.batching import filter_data, Sampler, Collator
from src.Name.neural.utils.schedules import make_schedule

from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from math import ceil
import sys


def train(config: TrainCfg, data_path: str, cast_to: str):
    logger = Logger(sys.stdout, './log.txt')
    sys.stdout = logger
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        print(f'Read {len(data)} files with {sum(len(hs) for _, _, hs in data)} holes.')

    data = list(filter_data(data,
                            max_scope_size=config['max_scope_size'],
                            max_db_index=config['model_config']['max_db_index'],
                            max_ast_len=config['max_ast_len']))
    print(f'Kept {len(data)} files with {sum(len(hs) for _, _, hs in data)} holes.')

    model = Trainer(config['model_config']).to(device(cast_to))
    train_sampler = Sampler([(name, scope, holes) for name, scope, holes in data if name in config['train_files']])
    dev_data = [(name, scope, holes) for name, scope, holes in data if name in config['dev_files']]
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, mode=model_config['mode'], cast_to=device(cast_to))

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
        train_epoch = model.train_epoch(
            epoch=map(collator, train_sampler.iter(
                batch_size_s=config['batch_size_s'],
                batch_size_h=config['batch_size_h'])),
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_every=config['backprop_every'])
        print(f'Train loss: {train_epoch["loss"]}')
        print(f'Train acc: {acc(train_epoch["predictions"])}')
        print('-' * 64)
        dev_epoch = model.eval_epoch(map(lambda x: collator([x]), dev_data))
        print(f'Dev loss: {dev_epoch["loss"]}')
        print(f'Dev acc: {acc(dev_epoch["predictions"])}')

        if dev_epoch['loss'] < best_loss:
            print('Saving...')
            model.save(f'./model.pt')
            best_loss = dev_epoch['loss']
        print('=' * 64 + '\n')


stdlib = [line for line in open('./data/stdlib.contents').read().split('\n')]

model_config: ModelCfg = {
    'mode':                 object,
    'depth':                8,
    'num_heads':            4,
    'dim':                  256,
    'atn_dim':              16,
    'share_depth_params':   True,
    'share_term_params':    True,
    'dropout_rate':         0.15,
    'max_db_index':         50
}

test_cfg: TrainCfg = {
    'model_config':         model_config,
    'num_epochs':           100,
    'warmup_epochs':        3,
    'warmdown_epochs':      90,
    'batch_size_s':         1,
    'batch_size_h':         64,
    'max_scope_size':       450,
    'max_ast_len':          1000,
    'max_lr':               5e-4,
    'min_lr':               1e-7,
    'backprop_every':       4,
    'train_files':          stdlib[:ceil(0.75 * len(stdlib))],
    'dev_files':            stdlib[ceil(0.75 * len(stdlib)):],
    'test_files':           [],
}


if __name__ == '__main__':
    train(test_cfg, './data/tokenized.p', 'cuda')
