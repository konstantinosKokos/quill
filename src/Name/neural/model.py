import pdb

import torch
from torch import Tensor
from .encoders import TypeEncoder
from torch.nn import Module, Linear, Sequential, Dropout


class Model(Module):
    def __init__(self, num_iters: int, num_layers: int,
                 dim: int, max_scope_size: int, max_db_index: int):
        super(Model, self).__init__()
        self.type_encoder = TypeEncoder(
            num_iters=num_iters,
            encoder_layers=num_layers,
            num_ops=3,
            num_leaves=4,
            max_db_index=max_db_index,
            max_scope_size=max_scope_size,
            dim=dim)
        self.lemma_predictor = Sequential(Linear(dim, dim // 2), Dropout(0.15), Linear(dim // 2, 1))

    def encode(self, dense_batch: Tensor, token_mask: Tensor) -> Tensor:
        return self.type_encoder.forward(dense_batch, token_mask)

    def predict_lemmas(self, token_reprs: Tensor, edge_index: Tensor) -> Tensor:
        type_reprs = token_reprs[:, :, 0].flatten(0, 1)
        scope_ids, goal_ids = edge_index
        return self.lemma_predictor(type_reprs[scope_ids] * type_reprs[goal_ids])

    def predict_masks(self, token_reprs: Tensor, lm_mask: Tensor,
                      batch_pointers: Tensor, tree_padding_mask: Tensor) -> Tensor:
        mask_reprs = token_reprs[lm_mask]
        type_reprs = token_reprs[:, :, 0]
        candidates = torch.index_select(type_reprs, 0, batch_pointers)
        candidate_mask = torch.index_select(tree_padding_mask, 0, batch_pointers)
        lm_preds = (mask_reprs.unsqueeze(-2) * candidates).sum(-1)
        lm_preds[~candidate_mask] = -1e10
        return lm_preds

    def forward(self, dense_batch: Tensor, token_mask: Tensor, edge_index: Tensor,
                lm_mask: Tensor, batch_pointers: Tensor, tree_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        token_reprs = self.encode(dense_batch, token_mask)
        lemmas = self.predict_lemmas(token_reprs, edge_index)
        lm_out = self.predict_masks(token_reprs, lm_mask, batch_pointers, tree_padding_mask)
        return lemmas.squeeze(-1), lm_out
