from torch import Tensor
from .embedding import TokenEncoder
from .encoders import TypeEncoder
from torch.nn import Module


class Model(Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.token_encoder = TokenEncoder(num_ops=3, num_leaves=4, dim=dim)
        self.type_encoder = TypeEncoder(dim, 4, 8, 8, False)

    def forward(self,
                token_types: Tensor,
                token_values: Tensor,
                tree_positions: Tensor,
                ground_positions: Tensor,
                padding_mask: Tensor) -> ...:

        token_embeddings = self.token_encoder(token_types, token_values, tree_positions, ground_positions)
        type_embeddings = self.type_encoder(token_embeddings, padding_mask)
        ...