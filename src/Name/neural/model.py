import pdb

import torch
from torch import Tensor
from .encoders import ScopeEncoder
from .utils import SwiGLU
from torch.nn import Module


class Model(Module):
    def __init__(self, num_layers: int, dim: int, max_db_index: int):
        super(Model, self).__init__()
        self.scope_encoder = ScopeEncoder(
            num_layers=num_layers,
            num_ops=3,
            num_leaves=4,
            max_db_index=max_db_index,
            dim=dim)
        self.lemma_predictor = SwiGLU(2 * dim, 4 * dim, 1)

    def encode(self, dense_batch: Tensor, token_mask: Tensor, tree_atn_mask: Tensor) -> Tensor:
        return self.scope_encoder.forward(dense_batch, token_mask, tree_atn_mask)

    def predict_lemmas(self, token_reprs: Tensor, edge_index: Tensor) -> Tensor:
        type_reprs = token_reprs[:, :, 0].flatten(0, 1)
        scope_ids, goal_ids = edge_index
        return self.lemma_predictor(torch.cat((type_reprs[scope_ids], type_reprs[goal_ids]), dim=-1))

    def predict_masks(self, token_reprs: Tensor, lm_mask: Tensor,
                      batch_pointers: Tensor, tree_padding_mask: Tensor) -> Tensor:
        mask_reprs = token_reprs[lm_mask]
        type_reprs = token_reprs[:, :, 0]
        candidates = torch.index_select(type_reprs, 0, batch_pointers)
        candidate_mask = torch.index_select(tree_padding_mask, 0, batch_pointers)
        lm_preds = (mask_reprs.unsqueeze(-2) * candidates).sum(-1)
        lm_preds[~candidate_mask] = -1e10
        return lm_preds

    def forward(self, dense_batch: Tensor, token_mask: Tensor, tree_atn_mask: Tensor,
                edge_index: Tensor, lm_mask: Tensor, batch_pointers: Tensor,
                tree_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        token_reprs = self.encode(dense_batch, token_mask, tree_atn_mask)
        lemmas = self.predict_lemmas(token_reprs, edge_index)
        lm_out = self.predict_masks(token_reprs, lm_mask, batch_pointers, tree_padding_mask)
        return lemmas.squeeze(-1), lm_out
