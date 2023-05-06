import pickle

import sys
sys.path.extend(['../'])

import torch
from src.Name.neural.batching import make_collator, Sampler
from src.Name.neural.training import TrainWrapper
from src.Name.neural.utils import make_schedule, binary_stats, macro_binary_stats
from torch import device as _device
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from math import ceil

with open('../data/tokenized.p', 'rb') as f:
    tokenized = pickle.load(f)



dim = 128
num_epochs = 100
num_layers = 8
batch_size = 2
backprop_every = 1
num_holes = 4
max_scope_size = 150
max_type_size = 100
max_db_index = 20
lm_chance = 0.1
device = _device('cuda')
dev_split = ceil(len(tokenized) * 0.25)


train_sampler = Sampler(tokenized[:-dev_split], max_type_size=max_type_size,
                        max_scope_size=max_scope_size, max_db_index=max_db_index)
dev_sampler = Sampler(tokenized[-dev_split:], max_type_size=max_type_size,
                      max_scope_size=max_scope_size, max_db_index=max_db_index)
collator = make_collator(device)

epoch_size = train_sampler.itersize(batch_size * backprop_every, num_holes)

model = TrainWrapper(num_layers=num_layers, dim=dim, max_db_index=max_db_index).to(device)


opt = AdamW(model.parameters(), lr=1, weight_decay=1e-02)
scheduler = LambdaLR(opt,
                     make_schedule(warmup_steps=3 * epoch_size,
                                   total_steps=100 * epoch_size,
                                   warmdown_steps=90 * epoch_size,
                                   max_lr=5e-4,
                                   min_lr=1e-7),
                     last_epoch=-1)

for epoch_id in range(num_epochs):
    epoch_lemma_loss, epoch_lm_loss = 0, 0
    epoch_lemma_preds, epoch_lemma_correct = [], []
    epoch_lm_hits, epoch_lm_total = 0, 0

    train_epoch = train_sampler.iter(batch_size, num_holes)
    model.train()

    with torch.autograd.set_detect_anomaly(True):
        for batch_id, batch in enumerate(train_epoch):
            collated = collator(batch, 0.1, 0.5)
            lemma_preds, gold_labels, (lm_hits, lm_total), lemma_loss, lm_loss = model.compute_losses(collated)
            loss = lemma_loss + lm_loss
            loss.backward()

            if (batch_id + 1) % backprop_every == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            epoch_lemma_loss += lemma_loss.item()
            epoch_lm_loss += lm_loss.item()
            epoch_lemma_preds += lemma_preds
            epoch_lemma_correct += gold_labels
            epoch_lm_hits += lm_hits
            epoch_lm_total += lm_total

    print('=' * 64)
    print(f'Epoch {epoch_id}')
    print(f'\tLM Loss: {epoch_lm_loss/epoch_lm_total:.2f}')
    print(f'\tLemma Loss: {epoch_lemma_loss/len(epoch_lemma_preds):.2f}')
    print(f'\tTotal: {epoch_lm_loss + epoch_lemma_loss:.2f}')
    print(f'\tLR: {scheduler.get_last_lr()[0]} (step: {scheduler.last_epoch})')
    tp, fn, tn, fp = binary_stats(epoch_lemma_preds, epoch_lemma_correct)
    accuracy, f1, prec, rec = macro_binary_stats(tp, fn, tn, fp)
    print('-' * 64)
    print(f'\tAccuracy: {epoch_lm_hits / epoch_lm_total:.2f} (LM, {epoch_lm_hits}/{epoch_lm_total})')
    print(f'\tAccuracy: {accuracy:.2f} (lemma)')
    print(f'\tPrecision: {prec:.2f}')
    print(f'\tRecall: {rec:.2f}')
    print(f'\tF1: {f1:.2f}')
    print('~' * 64)

    epoch_lemma_preds, epoch_lemma_correct = [], []
    epoch_dev_loss = 0
    model.eval()

    with torch.no_grad():
        for file in dev_sampler.filtered:
            lemma_preds, gold_labels, _, lemma_loss, _ = model.compute_losses(collator([file], 0.0, 1))
            epoch_dev_loss += lemma_loss.item()

            epoch_lemma_preds += lemma_preds
            epoch_lemma_correct += gold_labels

        print(f'\tDev Loss: {epoch_lemma_loss/len(epoch_lemma_preds)}')
        tp, fn, tn, fp = binary_stats(epoch_lemma_preds, epoch_lemma_correct)
        accuracy, f1, prec, rec = macro_binary_stats(tp, fn, tn, fp)
        print('-' * 64)
        print(f'\tDev Accuracy: {accuracy:.2f} (lemma)')
        print(f'\tDev Precision: {prec:.2f}')
        print(f'\tDev Recall: {rec:.2f}')
        print(f'\tDev F1: {f1:.2f}')
    print('\n')
