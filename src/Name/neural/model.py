import pdb

import torch
from torch import Tensor
from .embedding import TokenEncoder
from .encoders import TypeEncoder
from .utils import SwiGLU
from torch.nn import Module


class Model(Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.token_encoder = TokenEncoder(num_ops=3, num_leaves=4, dim=dim)
        self.type_encoder = TypeEncoder(dim, 4, 8, 8, False)
        self.predictor = SwiGLU(2 * dim, 4 * dim, 1)

    def embed(self,
              dense_batch: Tensor,
              token_mask: Tensor,
              type_mask: Tensor) -> Tensor:
        token_embeddings = self.token_encoder.forward(dense_batch)
        type_embeddings, token_embeddings = self.type_encoder.forward(token_embeddings, token_mask, type_mask)
        return type_embeddings

    def forward(self,
                dense_batch: Tensor,
                token_mask: Tensor,
                tree_mask: Tensor,
                edge_index: Tensor) -> Tensor:

        type_reprs = self.embed(dense_batch, token_mask, tree_mask).flatten(0, 1)
        scope_ids, goal_ids = edge_index
        features = torch.cat((type_reprs[scope_ids], type_reprs[goal_ids]), dim=-1)
        return self.predictor(features).squeeze(-1)