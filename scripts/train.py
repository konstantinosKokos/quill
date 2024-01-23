import json
import argparse
import pickle
import sys

sys.path.extend(['/home/kogkalk1/Projects/nagda/src'])

from Name.nn.training import TrainCfg, Trainer, Logger
from Name.nn.batching import filter_data, Sampler, Collator
from Name.nn.utils.schedules import make_schedule

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def train(
        config: TrainCfg,
        data_path: str,
        store_path: str,
        log_path: str,
        device: str):
    logger = Logger(sys.stdout, log_path)
    sys.stdout = logger
    print(train_cfg)

    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    files = list(filter_data(
        files=files,
        max_tokens=config['max_tokens']))
    print(f'Kept {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    train_files = [file for file in files if file.file.name in config['train_files']]
    dev_files = [file for file in files if file.file.name in config['dev_files']]
    print(f'Training on {len(train_files)} files with {sum(len(file.hole_asts) for file in train_files)} holes.')
    print(f'Evaluating on {len(dev_files)} files with {sum(len(file.hole_asts) for file in dev_files)} holes.')

    train_sampler = Sampler(train_files)
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, device=device, allow_self_loops=config['allow_self_loops'])

    model = Trainer(config['model_config']).to(device)
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
            model.save(store_path)
            best_ap = sum(dev_epoch.ap)
        print('=' * 64 + '\n')
    logger.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--data_path', type=str, help='Path to data file',
                        default='../data/tokenized.p')
    parser.add_argument('--config_path', type=str, help='Path to config file',
                        default='../data/config.json')
    parser.add_argument('--store_path', type=str, help='Where to store the trained model',
                        default='../data/model.pt')
    parser.add_argument('--log_path', type=str, help='Where to log results',
                        default='../data/log.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_cfg: TrainCfg = json.load(open(args.config_path, 'r'))
    train(
        config=train_cfg,
        data_path=args.data_path,
        store_path=args.store_path,
        log_path=args.log_path,
        device='cuda',
    )
