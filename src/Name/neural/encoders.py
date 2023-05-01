import pdb

from .utils import SwiGLU, RMSNorm, MHA
from .embedding import TokenEmbedder
from torch import Tensor
from torch.nn import Module, ModuleList, GRUCell, Dropout


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.mha = MHA(num_heads, dim, dropout_rate)
        self.ffn = SwiGLU(dim, dim, dim)
        self.ln_mha = RMSNorm(dim)
        self.ln_ffn = RMSNorm(dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        encoder_input = self.dropout(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, attention_mask)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        return self.ln_ffn(ffn_x)


class Encoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, dropout_rate: float):
        super(Encoder, self).__init__()
        self.layers = ModuleList([EncoderLayer(num_heads, dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        for layer in self.layers:
            encoder_input = layer(encoder_input, attention_mask)
        return encoder_input


class TypeEncoder(Module):
    def __init__(self, num_layers: int, num_ops: int, num_leaves: int,
                 dim: int, max_scope_size: int, max_db_index: int):
        super(TypeEncoder, self).__init__()
        self.embedder = TokenEmbedder(
            num_ops=num_ops,
            num_leaves=num_leaves,
            dim=dim,
            max_scope_size=max_scope_size,
            max_db_index=max_db_index)
        self.num_layers = num_layers
        self.intra_type = Encoder(4, 8, dim, 0.15)
        self.recurrence = GRUCell(dim, dim)

    def step(self, token_embeddings: Tensor, token_mask: Tensor,
             ref_mask: Tensor, ref_ids: Tensor) -> Tensor:
        (num_scopes, num_entries, _, _) = token_embeddings.shape

        token_embeddings = token_embeddings.flatten(0, 1)
        token_embeddings = self.intra_type.forward(token_embeddings, token_mask)
        token_embeddings = token_embeddings.unflatten(0, (num_scopes, num_entries))

        type_summaries = token_embeddings[:, :, 0]
        refs_h = token_embeddings[ref_mask]
        refs_x = type_summaries.flatten(0, 1)[ref_ids]
        ref_updates = self.recurrence.forward(refs_x, refs_h)
        token_embeddings[ref_mask] = ref_updates
        return token_embeddings

    def forward(self, dense_batch: Tensor, token_mask: Tensor) -> Tensor:
        token_embeddings, ref_mask, ref_ids = self.embedder.forward(dense_batch)
        for _ in range(self.num_layers):
            token_embeddings = self.step(token_embeddings, token_mask, ref_mask, ref_ids)
        return token_embeddings
