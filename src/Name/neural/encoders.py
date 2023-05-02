from .utils import RMSNorm, LinearMHA, ResidualFFN
from .embedding import TokenEmbedder
from torch import Tensor
from torch.nn import Module, ModuleList


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.mha_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.mha = LinearMHA(num_heads, dim, dim//8, dropout_rate)
        self.res_ffn = ResidualFFN(dim, 4 * dim, dropout_rate)

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        encoder_input = self.mha_norm(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, attention_mask)
        mha_x = encoder_input + mha_x
        ffn_x = self.res_ffn(mha_x)
        ffn_x = encoder_input + ffn_x
        return ffn_x


class Encoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, dropout_rate: float):
        super(Encoder, self).__init__()
        self.layers = ModuleList([EncoderLayer(num_heads, dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        for layer in self.layers:
            encoder_input = layer(encoder_input, attention_mask)
        return encoder_input


class TypeEncoder(Module):
    def __init__(self, num_iters: int, encoder_layers: int,
                 num_ops: int, num_leaves: int, dim: int,
                 max_scope_size: int, max_db_index: int):
        super(TypeEncoder, self).__init__()
        self.embedder = TokenEmbedder(
            num_ops=num_ops,
            num_leaves=num_leaves,
            dim=dim,
            max_scope_size=max_scope_size,
            max_db_index=max_db_index)
        self.num_iters = num_iters
        self.intra_type = Encoder(encoder_layers, 8, dim, 0.15)
        self.recurrence = ResidualFFN(dim, 4 * dim, 0.15)

    def step(self, token_embeddings: Tensor, token_mask: Tensor,
             ref_mask: Tensor, ref_ids: Tensor) -> Tensor:
        (num_scopes, num_entries, _, _) = token_embeddings.shape

        token_embeddings = token_embeddings.flatten(0, 1)
        token_embeddings = self.intra_type.forward(token_embeddings, token_mask)
        token_embeddings = token_embeddings.unflatten(0, (num_scopes, num_entries))

        type_summaries = token_embeddings[:, :, 0].flatten(0, 1)
        last_ref_reprs = token_embeddings[ref_mask]
        curr_ref_reprs = type_summaries[ref_ids]
        ref_updates = last_ref_reprs + curr_ref_reprs
        ref_updates = self.recurrence(ref_updates)
        token_embeddings[ref_mask] = ref_updates

        return token_embeddings

    def forward(self, dense_batch: Tensor, token_mask: Tensor) -> Tensor:
        token_embeddings, ref_mask, ref_ids = self.embedder.forward(dense_batch)
        for _ in range(self.num_iters):
            token_embeddings = self.step(token_embeddings, token_mask, ref_mask, ref_ids)
        return token_embeddings
