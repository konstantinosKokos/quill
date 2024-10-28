import json
import pickle
import sys

import torch

sys.path.extend(['/home/kogkalk1/Projects/nagda/src'])

from quill.nn.training import TrainCfg, Trainer
from quill.nn.batching import discard_empty, split_by_length, Collator
from quill.nn.utils.ranking import average_precision, rprecision


def evaluate(
        config: TrainCfg,
        data_path: str,
        model_paths: list[str],
        device: str,
        dev: bool = True,
        short: bool = True,
        long: bool = False):

    model = Trainer(config['model_config']).to(device)
    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')
    files = discard_empty(files)
    files = [f for f in files if f.file.name != 'foundation.morphisms-cospans']
    print(f'Of which {len(files)} have at least 1 hole.')

    files = [file for file in files if file.file.name in (config['dev_files'] if dev else config['train_files'])]
    match (short, long):
        case False, False:
            raise ValueError('Well, you must evaluate on something')
        case True, False:
            files, _ = split_by_length(files, config['max_tokens'])
        case False, True:
            _, files = split_by_length(files, config['max_tokens'])
        case True, True:
            pass
    print(f'Evaluating on {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    AP, RP, R1 = [], [], []
    with torch.no_grad():
        collator = Collator(pad_value=-1, device=device, allow_self_loops=config['allow_self_loops'])
        for model_path in model_paths:
            model.load(model_path, strict=True, map_location=device)
            model.eval()
            print(model_path)

            predictions, truths = model.infer_epoch(map(lambda x: collator([x]), files))
            aps = [average_precision(x, y) for x, y in zip(predictions, truths)]
            rps = [rprecision(x, y) for x, y in zip(predictions, truths)]
            r1s = [x[0] in y for x, y in zip(predictions, truths)]
            ap_stats = stats(aps)
            rp_stats = stats(rps)
            r1_stats = stats(r1s)
            AP.append(ap_stats[2]*100)
            RP.append(rp_stats[2]*100)
            R1.append(r1_stats[2]*100)
    print(stats(AP)[2])
    print(stats(RP)[2])
    print(stats(R1)[2])


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    mu = sum(xs) / len(xs)
    var = sum((x - mu)**2 for x in xs) ** 0.5
    return min(xs), max(xs), mu, var/len(xs)


if __name__ == '__main__':
    train_cfg: TrainCfg = json.load(open('/home/kokos/Projects/nagda/data/config.json', 'r'))
    evaluate(
        config=train_cfg,
        data_path='/home/kokos/Projects/nagda/data/tokenized.p',
        model_paths=[f'/home/kokos/Projects/nagda/data/rope{i}.pt' for i in range(0, 1)],
        long=True,
        short=False,
        device='cuda'
    )
