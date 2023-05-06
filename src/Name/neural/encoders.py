import pdb

from .utils import RMSNorm, LinearMHA, MHA, ResidualFFN
from .embedding import TokenEmbedder
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.functional import pad
from torch_scatter import scatter_mean

import torch

class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float, d_atn: int | None = None):
        super(EncoderLayer, self).__init__()
        self.mha_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        if d_atn is None:
            self.mha = MHA(num_heads, dim, dropout_rate)
        else:
            self.mha = LinearMHA(num_heads, dim, 32, dropout_rate)
        self.res_ffn = ResidualFFN(dim, 4 * dim, dropout_rate)

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        encoder_input = self.mha_norm(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, attention_mask)
        mha_x = encoder_input + mha_x
        ffn_x = self.res_ffn(mha_x)
        ffn_x = encoder_input + ffn_x
        return ffn_x


class TypeEncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float):
        super(TypeEncoderLayer, self).__init__()
        self.intra_type = EncoderLayer(num_heads, dim, dropout_rate, d_atn=32)
        self.inter_type = EncoderLayer(num_heads, dim, dropout_rate)
        self.recurrence = ResidualFFN(dim, 4 * dim, dropout_rate)

    def forward(self,
                token_embeddings: Tensor,
                token_mask: Tensor,
                tree_mask: Tensor,
                ref_mask: Tensor,
                ref_ids: Tensor) -> Tensor:
        (num_scopes, num_entries, num_tokens, _) = token_embeddings.shape

        # update tokens by their type context
        token_embeddings = token_embeddings.flatten(0, 1)
        token_embeddings = self.intra_type.forward(token_embeddings, token_mask)
        token_embeddings = token_embeddings.unflatten(0, (num_scopes, num_entries))

        # # update types by their scope context
        update_mask = torch.zeros_like(ref_mask, dtype=torch.bool)
        update_mask[:, :, 0] = True
        type_summaries = token_embeddings[update_mask].unflatten(0, (num_scopes, num_entries))
        type_updates = self.inter_type(type_summaries, tree_mask).flatten(0, 1)
        token_embeddings[update_mask] = type_updates

        # update references to types
        last_ref_reprs = token_embeddings[ref_mask]
        curr_ref_reprs = type_updates[ref_ids]
        ref_updates = last_ref_reprs + curr_ref_reprs
        ref_updates = self.recurrence(ref_updates)
        token_embeddings[ref_mask] = ref_updates

        return token_embeddings


class TypeEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, dropout_rate: float):
        super(TypeEncoder, self).__init__()
        self.layers = ModuleList([TypeEncoderLayer(num_heads, dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, token_embeddings: Tensor, token_mask: Tensor, tree_atn_mask: Tensor,
                ref_mask: Tensor, ref_ids: Tensor) -> Tensor:
        for layer in self.layers:
            token_embeddings = layer.forward(token_embeddings, token_mask, tree_atn_mask, ref_mask, ref_ids)
        return token_embeddings


class ScopeEncoder(Module):
    def __init__(self, num_layers: int, num_ops: int, num_leaves: int, dim: int, max_db_index: int):
        super(ScopeEncoder, self).__init__()
        self.embedder = TokenEmbedder(
            num_ops=num_ops,
            num_leaves=num_leaves,
            dim=dim,
            max_db_index=max_db_index)
        self.encoder = TypeEncoder(num_layers, 4, dim, 0.1)

    def forward(self, dense_batch: Tensor, token_mask: Tensor, tree_atn_mask: Tensor) -> Tensor:
        token_embeddings, ref_mask, ref_ids = self.embedder.forward(dense_batch)
        return self.encoder.forward(token_embeddings, token_mask, tree_atn_mask, ref_mask, ref_ids)
