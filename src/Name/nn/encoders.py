import torch
from torch.nn import Module, ModuleList
from torch import Tensor

from .batching import BatchedASTs

from .utils.modules import EncoderLayer
from .embedding import TokenEmbedding


class FileEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, head_dim: int, dropout_rate: float):
        super(FileEncoder, self).__init__()
        self.dim = dim
        self.term_encoder = TermEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            head_dim=head_dim,
            dropout_rate=dropout_rate)
        self.embedding = TokenEmbedding(dim=dim)

    def forward(self,
                scope_asts: BatchedASTs,
                scope_sort: Tensor,
                hole_asts: BatchedASTs) -> tuple[Tensor, Tensor]:

        scope_features = self.embedding.forward(scope_asts.tokens.permute(2, 0, 1))
        hole_features = self.embedding.forward(hole_asts.tokens.permute(2, 0, 1))

        scope_reprs = torch.zeros(
            scope_asts.num_trees,
            self.dim,
            dtype=scope_features.dtype,
            device=scope_sort.device)

        for rank in scope_sort.unique(sorted=True):
            rank_mask = scope_sort.eq(rank)
            padding_mask = scope_asts.padding_mask[rank_mask]
            max_seq_len = padding_mask.sum(-1).max().item()

            scope_reprs[rank_mask] = self.term_encoder.forward(
                dense_features=scope_features[rank_mask][:, :max_seq_len],
                padding_mask=padding_mask[:, :max_seq_len],
                reference_mask=scope_asts.reference_mask[rank_mask][:, :max_seq_len],
                reference_ids=scope_asts.tokens[rank_mask][scope_asts.reference_mask[rank_mask]][:, 1],
                reference_storage=scope_reprs
            )
        hole_reprs = self.term_encoder.forward(
            dense_features=hole_features,
            padding_mask=hole_asts.padding_mask,
            reference_mask=hole_asts.reference_mask,
            reference_ids=hole_asts.tokens[hole_asts.reference_mask][:, 1],
            reference_storage=scope_reprs
        )
        return scope_reprs, hole_reprs


class TermEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, head_dim: int, dropout_rate: float):
        super(TermEncoder, self).__init__()
        self.encoder = ModuleList([
            EncoderLayer(num_heads=num_heads,
                         dim=dim,
                         head_dim=head_dim,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)])

    def forward(self,
                dense_features: Tensor,
                padding_mask: Tensor,
                reference_mask: Tensor,
                reference_ids: Tensor,
                reference_storage: Tensor) -> Tensor:
        dense_features[reference_mask] = reference_storage[reference_ids]  # todo : make fancy

        layer: EncoderLayer
        for layer in self.encoder:
            dense_features = layer.forward(dense_features, padding_mask)
        return dense_features[:, 0]
