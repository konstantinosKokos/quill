import pickle

from src.Name.neural.train import TrainCfg, Trainer, macro_binary_stats, binary_stats
from src.Name.neural.batching import filter_data, Sampler, Collator
from src.Name.neural.utils.schedules import make_schedule

from torch import device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def train(config: TrainCfg, data_path: str, cast_to: str):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        print(f'Read {len(data)} files.')

    data = list(filter_data(data,
                            max_scope_size=config['max_scope_size'],
                            max_db_index=config['model_config']['max_db_index'],
                            max_ast_len=config['max_ast_len']))
    print(f'Kept {len(data)} files.')

    model = Trainer(config['model_config']).to(device(cast_to))
    train_sampler = Sampler([(name, scope, holes) for name, scope, holes in data if name in config['train_files']])
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, cast_to=device(cast_to))

    optimizer = AdamW(params=model.parameters(), lr=1, weight_decay=1e-02)
    schedule = make_schedule(warmup_steps=config['warmup_epochs'] * epoch_size,
                             warmdown_steps=config['warmdown_epochs'] * epoch_size,
                             max_lr=config['max_lr'],
                             min_lr=config['min_lr'],
                             total_steps=config['num_epochs'] * epoch_size)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=schedule, last_epoch=-1)

    for epoch in range(config['num_epochs']):
        epoch_stats = model.train_epoch(
            epoch=map(collator, train_sampler.iter(
                batch_size_s=config['batch_size_s'],
                batch_size_h=config['batch_size_h'])),
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_every=config['backprop_every'])
        print(macro_binary_stats(*binary_stats(epoch_stats['predictions'], epoch_stats['truth'])))


test_cfg: TrainCfg = {
    'model_config':     {
        'depth':                6,
        'num_heads':            4,
        'dim':                  128,
        'atn_dim':              16,
        'share_depth_params':   True,
        'share_term_params':    True,
        'dropout_rate':         0.1,
        'max_db_index':         50
    },
    'num_epochs':       100,
    'warmup_epochs':    3,
    'warmdown_epochs':  90,
    'batch_size_s':     2,
    'batch_size_h':     4,
    'max_scope_size':   150,
    'max_ast_len':      300,
    'max_lr':           5e-4,
    'min_lr':           1e-7,
    'backprop_every':   2,
    'train_files':      [line for line in open('./data/stdlib.contents').read().split('\n')],
    'dev_files':        [],
    'test_files':       [],
}
