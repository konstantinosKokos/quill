import torch
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.functional import binary_cross_entropy_with_logits

from .attention import lin_atn_fn, atn_fn


def swish(x: Tensor, b: int = 1) -> Tensor:
    gate = torch.sigmoid(b * x)
    return x * gate


def focal_loss(inputs: Tensor, targets: Tensor, gamma: float) -> Tensor:
    alpha = sum(targets)/sum(~targets)
    bce_loss = binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none', pos_weight=1/alpha)
    probs = inputs.sigmoid()
    distance = torch.where(targets, 1 - probs, probs)
    loss = (distance ** gamma) * bce_loss
    return loss


def dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    probs = inputs.sigmoid()
    soft_tp = inputs[targets == 1].sum()
    soft_f = torch.where(targets, 1 - probs, probs).sum()
    return - 2 * soft_tp / (2 * soft_tp + soft_f)


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int, output_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim, bias=False)
        self.v = Linear(input_dim, interm_dim, bias=False)
        self.w_out = Linear(interm_dim, output_dim, bias=False)

    def forward(self, x: Tensor, gate: Tensor | None) -> Tensor:
        interm = self.w_in(x if gate is None else x)
        interm = swish(interm) * self.v(x)
        return self.w_out(interm)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class MHA(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float = 0.1):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.q_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.k_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.v_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = atn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)


class LinearMHA(Module):
    def __init__(self, num_heads: int, dim: int, d_atn: int, dropout_rate: float = 0.1):
        super(LinearMHA, self).__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads')
        self.q_transformation = Linear(in_features=dim, out_features=d_atn * num_heads, bias=False)
        self.k_transformation = Linear(in_features=dim, out_features=d_atn * num_heads, bias=False)
        self.v_transformation = Linear(in_features=dim, out_features=dim, bias=False)
        self.wo = Linear(in_features=dim, out_features=dim, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = lin_atn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)


class ResidualFFN(Module):
    def __init__(self, dim: int, intermediate: int, dropout_rate: float):
        super(ResidualFFN, self).__init__()
        self.pre_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate, dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor, gate: Tensor | None = None) -> Tensor:
        norm = self.pre_norm(x)
        ffn = self.ffn(norm, gate)
        return ffn + norm


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float, atn_dim: int | None):
        super(EncoderLayer, self).__init__()
        self.mha_norm = RMSNorm(dim)
        if atn_dim is None:
            self.mha = MHA(num_heads, dim, dropout_rate)
        else:
            self.mha = LinearMHA(num_heads, dim, atn_dim, dropout_rate)
        self.res_ffn = ResidualFFN(dim, 4 * dim, dropout_rate)

    def forward(self, encoder_input: Tensor, attention_mask: Tensor) -> Tensor:
        encoder_input = self.mha_norm(encoder_input)
        mha_x = self.mha(encoder_input, encoder_input, encoder_input, attention_mask)
        mha_x = encoder_input + mha_x
        ffn_x = self.res_ffn(mha_x)
        ffn_x = encoder_input + ffn_x
        return ffn_x
