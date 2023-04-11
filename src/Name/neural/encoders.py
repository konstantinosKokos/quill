from .utils import SwiGLU, RMSNorm, MultiHeadAttention
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Dropout


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mha = MultiHeadAttention(num_heads, dim, dropout_rate)
        self.ffn = SwiGLU(dim, 2 * dim, dim)
        self.ln_mha = RMSNorm(dim)
        self.ln_ffn = RMSNorm(dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, encoder_input: Tensor, padding_mask: Tensor) -> Tensor:
        encoder_input = self.dropout(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, padding_mask)
        mha_x = self.dropout(mha_x)
        mha_x = encoder_input + mha_x
        mha_x = self.ln_mha(mha_x)

        ffn_x = self.ffn(mha_x)
        ffn_x = self.dropout(ffn_x)
        ffn_x = ffn_x + mha_x
        return self.ln_ffn(ffn_x)


class TypeEncoder(Module):
    def __init__(self, dim: int, intra_heads: int, inter_heads: int, num_iters: int, parameter_sharing: bool):
        super(TypeEncoder, self).__init__()
        self.num_iters = num_iters
        self.parameter_sharing = parameter_sharing
        num_modules = 1 if parameter_sharing else num_iters

        self.intra_type = ModuleList([EncoderLayer(intra_heads, dim, 0.1) for _ in range(num_modules)])
        self.inter_type = ModuleList([EncoderLayer(inter_heads, dim, 0.1) for _ in range(num_modules)])
        self.ffn = ModuleList([Sequential(SwiGLU(dim, 2 * dim, dim), RMSNorm(dim)) for _ in range(num_modules)])

    def step(self, token_embeddings: Tensor, token_mask: Tensor, type_mask: Tensor, step: int) -> Tensor:
        num_scopes, num_entries, num_tokens, dim = token_embeddings.shape

        if self.parameter_sharing:
            step = 0

        token_embeddings = token_embeddings.flatten(0, 1)
        token_embeddings = self.intra_type[step](token_embeddings, token_mask)
        token_embeddings = token_embeddings.unflatten(0, (num_scopes, num_entries))

        seeds = token_embeddings[:, :, 0]
        type_embeddings = self.inter_type[step](seeds, type_mask)
        seeds = self.ffn[step](seeds + type_embeddings)
        token_embeddings[:, :, 0] = seeds

        return token_embeddings

    def forward(self, token_embeddings: Tensor, padding_mask: Tensor) -> Tensor:
        num_scopes, num_entries, num_tokens, dim = token_embeddings.shape
        token_mask = padding_mask.flatten(0, 1).unsqueeze(-2).repeat(1, num_tokens, 1)
        type_mask = padding_mask.any(-1).unsqueeze(-2).repeat(1, num_entries, 1)

        for step in range(self.num_iters):
            token_embeddings = self.step(token_embeddings, token_mask, type_mask, step)
        return token_embeddings[:, :, 0]
