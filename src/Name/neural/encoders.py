import torch
from torch.nn import Module, ModuleList, Parameter
from torch import Tensor

from .utils.modules import EncoderLayer, ResidualFFN, SwiGLU, RMSNorm


class ReferenceUpdater(Module):
    def __init__(self, dim: int, dropout_rate: float):
        super(ReferenceUpdater, self).__init__()
        self.fusion_nn = ResidualFFN(dim=dim, intermediate=4 * dim, dropout_rate=dropout_rate)

    def forward(self,
                token_embeddings: Tensor,
                reference_mask: Tensor,
                reference_ids: Tensor,
                reference_embeddings: Tensor) -> Tensor:
        gate = token_embeddings[reference_mask]
        ctx = reference_embeddings.flatten(0, 1)[reference_ids]
        update = self.fusion_nn.forward(x=ctx, gate=gate)
        token_embeddings[reference_mask] = update
        return token_embeddings


class TermEncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, atn_dim: int, dropout_rate: float):
        super(TermEncoderLayer, self).__init__()
        self.encoder = EncoderLayer(num_heads=num_heads, dim=dim, atn_dim=atn_dim, dropout_rate=dropout_rate)
        self.ref_fuse = ReferenceUpdater(dim=dim, dropout_rate=dropout_rate)

    def forward(self,
                token_embeddings: Tensor,
                token_mask: Tensor,
                reference_mask: Tensor,
                reference_ids: Tensor,
                reference_embeddings: Tensor) -> Tensor:
        (num_files, num_entries, num_tokens, _) = token_embeddings.shape
        token_embeddings = token_embeddings.flatten(0, 1)
        token_embeddings = self.encoder.forward(encoder_input=token_embeddings,
                                                attention_mask=token_mask).unflatten(0, (num_files, num_entries))
        token_embeddings = self.ref_fuse.forward(token_embeddings=token_embeddings,
                                                 reference_mask=reference_mask,
                                                 reference_ids=reference_ids,
                                                 reference_embeddings=reference_embeddings)
        return token_embeddings


class EntryEncoderLayer(Module):
    def __init__(self, share_params: bool, num_heads: int, dim: int, atn_dim: int, dropout_rate: float):
        super(EntryEncoderLayer, self).__init__()
        self.share_params = share_params
        encoder_kwargs = {'num_heads': num_heads, 'dim': dim, 'atn_dim': atn_dim, 'dropout_rate': dropout_rate}
        self.type_encoder = TermEncoderLayer(**encoder_kwargs)
        self.def_encoder = self.type_encoder if self.share_params else TermEncoderLayer(**encoder_kwargs)
        self.entry_encoder = SwiGLU(dim, 4 * dim, dim)
        self.entry_norm = RMSNorm(dim)

    def forward(self,
                type_token_embeddings: Tensor,
                type_token_mask: Tensor,
                def_token_embeddings: Tensor,
                def_token_mask: Tensor,
                type_token_ref_mask: Tensor,
                type_token_ref_ids: Tensor,
                def_token_ref_mask: Tensor,
                def_token_ref_ids: Tensor,
                reference_embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        type_token_embeddings = self.type_encoder.forward(
            token_embeddings=type_token_embeddings,
            token_mask=type_token_mask,
            reference_mask=type_token_ref_mask,
            reference_ids=type_token_ref_ids,
            reference_embeddings=reference_embeddings)
        def_token_embeddings = self.def_encoder.forward(
            token_embeddings=def_token_embeddings,
            token_mask=def_token_mask,
            reference_mask=def_token_ref_mask,
            reference_ids=def_token_ref_ids,
            reference_embeddings=reference_embeddings)
        entry_embeddings = self.entry_encoder.forward(x=def_token_embeddings[:, :, 0],
                                                      gate=type_token_embeddings[:, :, 0])
        entry_embeddings = entry_embeddings + self.entry_norm(reference_embeddings)
        return entry_embeddings, type_token_embeddings, def_token_embeddings


class FileEncoderLayer(Module):
    def __init__(self,
                 share_term_params: bool,
                 num_heads: int,
                 dim: int,
                 atn_dim: int,
                 dropout_rate: float):
        super(FileEncoderLayer, self).__init__()
        self.share_term_params = share_term_params
        self.encoder = EntryEncoderLayer(
            share_params=share_term_params, dim=dim,  dropout_rate=dropout_rate,
            atn_dim=atn_dim, num_heads=num_heads)

    def forward(self,
                scope_type_embeddings: Tensor,
                scope_type_mask: Tensor,
                scope_def_embeddings: Tensor,
                scope_def_mask: Tensor,
                scope_type_ref_mask: Tensor,
                scope_type_ref_ids: Tensor,
                scope_def_ref_mask: Tensor,
                scope_def_ref_ids: Tensor,
                hole_type_embeddings: Tensor,
                hole_type_mask: Tensor,
                hole_type_ref_mask: Tensor,
                hole_type_ref_ids: Tensor,
                scope_embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (scope_embeddings, scope_type_embeddings, scope_def_embeddings) = self.encoder.forward(
            type_token_embeddings=scope_type_embeddings,
            type_token_mask=scope_type_mask,
            def_token_embeddings=scope_def_embeddings,
            def_token_mask=scope_def_mask,
            type_token_ref_mask=scope_type_ref_mask,
            type_token_ref_ids=scope_type_ref_ids,
            def_token_ref_mask=scope_def_ref_mask,
            def_token_ref_ids=scope_def_ref_ids,
            reference_embeddings=scope_embeddings)
        hole_type_embeddings = self.encoder.type_encoder.forward(
            token_embeddings=hole_type_embeddings,
            token_mask=hole_type_mask,
            reference_mask=hole_type_ref_mask,
            reference_ids=hole_type_ref_ids,
            reference_embeddings=scope_embeddings)
        return scope_embeddings, scope_type_embeddings, scope_def_embeddings, hole_type_embeddings


class FileEncoder(Module):
    def __init__(self,
                 share_depth_params: bool,
                 share_term_params: bool,
                 depth: int,
                 num_heads: int,
                 dim: int,
                 atn_dim: int,
                 dropout_rate: float):
        super(FileEncoder, self).__init__()
        self.share_depth_params = share_depth_params
        self.share_term_params = share_term_params
        self.depth = depth
        if not self.share_depth_params:
            self.layers = ModuleList([FileEncoderLayer(
                share_term_params=share_term_params, dim=dim, dropout_rate=dropout_rate,
                atn_dim=atn_dim, num_heads=num_heads)
                for _ in range(depth)])
        else:
            self.layers = ModuleList([FileEncoderLayer(
                share_term_params=share_term_params, dim=dim, dropout_rate=dropout_rate,
                atn_dim=atn_dim, num_heads=num_heads)])
        self.init_seed = Parameter(torch.rand(dim), requires_grad=True)

    def forward(self,
                scope_type_embeddings: Tensor,
                scope_type_mask: Tensor,
                scope_def_embeddings: Tensor,
                scope_def_mask: Tensor,
                scope_type_ref_mask: Tensor,
                scope_type_ref_ids: Tensor,
                scope_def_ref_mask: Tensor,
                scope_def_ref_ids: Tensor,
                hole_type_embeddings: Tensor,
                hole_type_mask: Tensor,
                hole_type_ref_mask: Tensor,
                hole_type_ref_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scope_embeddings = self.init_seed.expand_as(scope_type_embeddings[:, :, 0])
        for depth in range(self.depth):
            layer_idx = depth if not self.share_depth_params else 0
            (scope_embeddings, scope_type_embeddings, scope_def_embeddings, hole_type_embeddings) = \
                self.layers[layer_idx].forward(
                    scope_type_embeddings=scope_type_embeddings,
                    scope_type_mask=scope_type_mask,
                    scope_def_embeddings=scope_def_embeddings,
                    scope_def_mask=scope_def_mask,
                    scope_type_ref_mask=scope_type_ref_mask,
                    scope_type_ref_ids=scope_type_ref_ids,
                    scope_def_ref_mask=scope_def_ref_mask,
                    scope_def_ref_ids=scope_def_ref_ids,
                    hole_type_embeddings=hole_type_embeddings,
                    hole_type_mask=hole_type_mask,
                    hole_type_ref_mask=hole_type_ref_mask,
                    hole_type_ref_ids=hole_type_ref_ids,
                    scope_embeddings=scope_embeddings)
            return scope_embeddings, scope_type_embeddings, scope_def_embeddings, hole_type_embeddings
