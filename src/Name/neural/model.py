from torch import Tensor
from .embedding import TokenEncoder
from .encoders import TypeEncoder, CrossEncoder
from torch.nn import Module, Linear


class Model(Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.token_encoder = TokenEncoder(num_ops=3, num_leaves=4, dim=dim)
        self.type_encoder = TypeEncoder(dim, 4, 8, 8, False)
        self.cross_encoder = CrossEncoder(num_heads=4, dim=dim)
        self.predictor = Linear(dim, 1)

    def batch_embed(self,
                    token_types: Tensor,
                    token_values: Tensor,
                    token_positions: Tensor,
                    tree_positions: Tensor,
                    padding_mask: Tensor) -> Tensor:
        token_embeddings = self.token_encoder(token_types, token_values, token_positions, tree_positions)
        return self.type_encoder(token_embeddings, padding_mask)

    def forward(self,
                scopes: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                contexts: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
                routing: Tensor,
                padding: Tensor) -> Tensor:
        """

        :param scopes: the collection of tensors for the scopes, each of batch shape S x M
        :param contexts: the collection of tensors for the contexts, each of batch shape C x N
        :param routing: the routing tensor, of shape C x S
        :param padding: the cross-padding mask, of shape C x M x N
        :return: predictions for each context over its routed scope, of shape C x M
        """
        scope_embeddings = self.batch_embed(*scopes)
        context_embeddings = self.batch_embed(*contexts)
        cross_embeddings = self.cross_encoder(scope_embeddings, context_embeddings, routing, padding)
        # todo: mask out positive predictions over masked scope entries
        return self.predictor(cross_embeddings)
