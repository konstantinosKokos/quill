import torch
from torch import Tensor
from torch.nn import Module, Linear
from typing import TypedDict

from .utils.modules import focal_loss
from .encoders import FileEncoder
from .embedding import TokenEmbedding
from .batching import Batch


class ModelCfg(TypedDict):
    depth: int
    num_heads: int
    dim: int
    atn_dim: int
    share_depth_params: bool
    share_term_params: bool
    dropout_rate: float
    max_db_index: int


class Model(Module):
    def __init__(self, config: ModelCfg):
        super(Model, self).__init__()
        self.file_encoder = FileEncoder(
            share_depth_params=config['share_depth_params'],
            share_term_params=config['share_term_params'],
            num_heads=config['num_heads'],
            dim=config['dim'],
            atn_dim=config['atn_dim'],
            dropout_rate=config['dropout_rate'],
            depth=config['depth'])
        self.embedding = TokenEmbedding(dim=config['dim'], max_db_index=config['max_db_index'])
        self.lemma_predictor = Linear(config['dim'], 1, bias=False)

    def encode(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scope_type_embeddings = self.embedding.forward(batch.scope_types.tokens)
        scope_def_embeddings = self.embedding.forward(batch.scope_definitions.tokens)
        hole_type_embeddings = self.embedding.forward(batch.hole_types.tokens)

        return self.file_encoder.forward(
            scope_type_embeddings=scope_type_embeddings,
            scope_type_mask=batch.scope_types.token_mask,
            scope_def_embeddings=scope_def_embeddings,
            scope_def_mask=batch.scope_definitions.token_mask,
            scope_type_ref_mask=batch.scope_types.reference_mask,
            scope_type_ref_ids=batch.scope_types.reference_ids,
            scope_def_ref_mask=batch.scope_definitions.reference_mask,
            scope_def_ref_ids=batch.scope_definitions.reference_ids,
            hole_type_embeddings=hole_type_embeddings,
            hole_type_mask=batch.hole_types.token_mask,
            hole_type_ref_mask=batch.hole_types.reference_mask,
            hole_type_ref_ids=batch.hole_types.reference_ids)

    def predict_lemmas(self, scope_reprs: Tensor, hole_reprs: Tensor, edge_index: Tensor) -> Tensor:
        source_index, target_index = edge_index
        sources = scope_reprs.flatten(0, 1)[source_index]
        targets = hole_reprs.flatten(0, 1)[target_index]
        return self.lemma_predictor(sources * targets).squeeze(-1)

    def compute_loss(self, batch: Batch) -> tuple[list[bool], list[bool], Tensor]:
        scope_reprs, _, _, hole_reprs = self.encode(batch)
        predictions = self.predict_lemmas(scope_reprs=scope_reprs,
                                          hole_reprs=hole_reprs[:, :, 0],
                                          edge_index=batch.edge_index)
        loss = focal_loss(predictions, batch.lemmas, gamma=2.)
        return (predictions.sigmoid().round().cpu().bool().tolist(),
                batch.lemmas.cpu().tolist(),
                loss)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location), strict=True)

