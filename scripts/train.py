import pdb
import pickle

import torch
from src.Name.neural.batching import make_collator, Sampler
from src.Name.neural.model import Model
from src.Name.neural.utils import make_schedule
from torch import device as _device
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

with open('../data/tokenized.p', 'rb') as f:
    tokenized = pickle.load(f)

device = _device('cpu')
sampler = Sampler(tokenized, 150, 50, 20)
collator = make_collator(device, lm_chance=0.1)

dim = 768
num_epochs = 100
batch_size = 1
backprop_every = 1
num_holes = 4

epoch_size = sampler.itersize(batch_size * backprop_every, num_holes)

model = Model(4, 128, 150, 20).to(device)

lemma_loss_fn = BCEWithLogitsLoss(reduction='sum')
lm_loss_fn = CrossEntropyLoss(reduction='sum')

opt = AdamW(model.parameters(), lr=1)
scheduler = LambdaLR(opt,
                     make_schedule(warmup_steps=3 * epoch_size,
                                   total_steps=100 * epoch_size,
                                   warmdown_steps=90 * epoch_size,
                                   max_lr=1e-3,
                                   min_lr=1e-6),
                     last_epoch=-1)

for epoch_id in range(num_epochs):
    epoch_lemma_loss, epoch_lm_loss = 0, 0
    epoch_lemma_preds, epoch_lemma_correct = [], []
    epoch_lm_preds, epoch_lm_correct = 0, 0

    epoch = sampler.iter(batch_size, num_holes)

    for batch_id, batch in enumerate(epoch):
        dense_batch, token_mask, tree_mask, edge_index, gold_labels, lm_mask, mask_values, batch_pts = collator(batch)
        lemma_preds, lm_preds = model.forward(dense_batch, token_mask, edge_index, lm_mask, batch_pts, tree_mask)
        lemma_loss = lemma_loss_fn(lemma_preds.squeeze(-1), gold_labels.float())
        lm_loss = lm_loss_fn(lm_preds, mask_values)
        loss = lemma_loss + lm_loss
        loss.backward()

        if (batch_id + 1) % backprop_every == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

        epoch_lemma_loss += lemma_loss.item()
        epoch_lm_loss += lm_loss.item()
        epoch_lemma_preds += lemma_preds.sigmoid().round().bool().cpu().tolist()
        epoch_lemma_correct += gold_labels.bool().cpu().tolist()
        epoch_lm_preds += len(mask_values)
        epoch_lm_correct += (lm_preds.argmax(dim=-1).eq(mask_values)).sum().item()

    print('=' * 64)
    print(f'Epoch {epoch_id}')
    print(f'\tLM Loss: {epoch_lm_loss}')
    print(f'\tLemma Loss: {epoch_lemma_loss}')
    print(f'\tTotal: {epoch_lm_loss + epoch_lemma_loss}')
    print('-' * 64)
    print(f'\tLR: {scheduler.get_last_lr()} (step: {scheduler.last_epoch}')
    tp = sum([x == y for x, y in zip(epoch_lemma_preds, epoch_lemma_correct) if y])
    fn = sum([x != y for x, y in zip(epoch_lemma_preds, epoch_lemma_correct) if y])
    tn = sum([x == y for x, y in zip(epoch_lemma_preds, epoch_lemma_correct) if not y])
    fp = sum([x != y for x, y in zip(epoch_lemma_preds, epoch_lemma_correct) if not y])
    acc = sum(x == y for x, y in zip(epoch_lemma_preds, epoch_lemma_correct))/len(epoch_lemma_preds)
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    print(f'\tAccuracy: {acc} (lemma) // {epoch_lm_correct/epoch_lm_preds} (LM)')
    print(f'\tPrecision: {prec}')
    print(f'\tRecall: {rec}')
    print(f'\tF1: {f1}')
    print(f'\t\t {tp=} // {fn=} // {tn=} // {fp=}')
