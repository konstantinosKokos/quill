from .utils.modules import RMSNorm, LinearMHA, MHA, ResidualFFN
from .embedding import TokenEmbedder
from .batching import Batch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.distributions import Gumbel
import torch


class EncoderLayer(Module):
    def __init__(self, num_heads: int, dim: int, dropout_rate: float, d_atn: int | None = None):
        super(EncoderLayer, self).__init__()
        self.mha_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        if d_atn is None:
            self.mha = MHA(num_heads, dim, dropout_rate)
        else:
            self.mha = LinearMHA(num_heads, dim, d_atn, dropout_rate)
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
        self.ref_update = ResidualFFN(dim, 4 * dim, dropout_rate)

    def forward(self,
                scope_token_embeddings: Tensor, scope_token_mask: Tensor, scope_tree_mask: Tensor,
                scope_ref_mask: Tensor, scope_ref_ids: Tensor, goal_token_embeddings: Tensor,
                goal_token_mask: Tensor, goal_ref_mask: Tensor, goal_ref_ids: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        (num_scopes, num_entries, num_scope_tokens, _) = scope_token_embeddings.shape
        (_, num_goals, num_goal_tokens, _) = goal_token_embeddings.shape

        # update tokens by their type context (scopes)
        scope_token_embeddings = scope_token_embeddings.flatten(0, 1)
        scope_token_embeddings = self.intra_type.forward(scope_token_embeddings, scope_token_mask)
        scope_token_embeddings = scope_token_embeddings.unflatten(0, (num_scopes, num_entries))

        # update tokens by their type context (goals)
        goal_token_embeddings = goal_token_embeddings.flatten(0, 1)
        goal_token_embeddings = self.intra_type.forward(goal_token_embeddings, goal_token_mask)
        goal_token_embeddings = goal_token_embeddings.unflatten(0, (num_scopes, num_goals))

        # update lemmas by their term embeddings
        update_mask = torch.zeros_like(scope_ref_mask, dtype=torch.bool)
        update_mask[:, :, 0] = True
        type_updates = scope_token_embeddings[update_mask] + noise.flatten(0, 1)
        scope_token_embeddings[update_mask] = type_updates

        # update references to types (scopes)
        scope_ref_reprs = scope_token_embeddings[scope_ref_mask]
        scope_ref_updates = type_updates[scope_ref_ids]
        scope_ref_updates = scope_ref_reprs + scope_ref_updates
        scope_ref_updates = self.ref_update(scope_ref_updates)
        scope_token_embeddings[scope_ref_mask] = scope_ref_updates

        # update references to types (goals)
        goal_ref_reprs = goal_token_embeddings[goal_ref_mask]
        goal_ref_updates = type_updates[goal_ref_ids]
        goal_ref_updates = goal_ref_reprs + goal_ref_updates
        goal_ref_updates = self.ref_update(goal_ref_updates)
        goal_token_embeddings[goal_ref_mask] = goal_ref_updates

        return scope_token_embeddings, goal_token_embeddings


class TypeEncoder(Module):
    def __init__(self, num_layers: int, num_heads: int, dim: int, dropout_rate: float):
        super(TypeEncoder, self).__init__()
        self.layers = ModuleList([TypeEncoderLayer(num_heads, dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, scope_token_embeddings: Tensor, scope_token_mask: Tensor, scope_tree_mask: Tensor,
                scope_ref_mask: Tensor, scope_ref_ids: Tensor, goal_token_embeddings: Tensor,
                goal_token_mask: Tensor, goal_ref_mask: Tensor, goal_ref_ids: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            scope_token_embeddings, goal_token_embeddings = layer.forward(
                scope_token_embeddings=scope_token_embeddings,
                scope_token_mask=scope_token_mask,
                scope_tree_mask=scope_tree_mask,
                scope_ref_mask=scope_ref_mask,
                scope_ref_ids=scope_ref_ids,
                goal_token_embeddings=goal_token_embeddings,
                goal_token_mask=goal_token_mask,
                goal_ref_mask=goal_ref_mask,
                goal_ref_ids=goal_ref_ids,
                noise=noise)
        return scope_token_embeddings, goal_token_embeddings


class ScopeEncoder(Module):
    def __init__(self, num_layers: int, num_ops: int, num_leaves: int, dim: int, max_db_index: int):
        super(ScopeEncoder, self).__init__()
        self.embedder = TokenEmbedder(
            num_ops=num_ops,
            num_leaves=num_leaves,
            dim=dim,
            max_db_index=max_db_index)
        self.encoder = TypeEncoder(num_layers, 4, dim, 0.1)
        self.noise_gen = Gumbel(0., 0.1)

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        scope_token_embeddings = self.embedder.forward(batch.dense_scopes, batch.lm_mask)
        goal_token_embeddings = self.embedder.forward(batch.dense_goals, None)
        noise = self.noise_gen.sample(scope_token_embeddings[:, :, 0].shape).to(scope_token_embeddings.device)
        return self.encoder.forward(
            scope_token_embeddings=scope_token_embeddings,
            scope_token_mask=batch.scope_token_mask,
            scope_tree_mask=batch.scope_tree_mask,
            scope_ref_mask=batch.scope_ref_mask,
            scope_ref_ids=batch.scope_ref_ids,
            goal_token_embeddings=goal_token_embeddings,
            goal_token_mask=batch.goal_token_mask,
            goal_ref_mask=batch.goal_ref_mask,
            goal_ref_ids=batch.goal_ref_ids,
            noise=noise)
