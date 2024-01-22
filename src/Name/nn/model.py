import torch
from torch import Tensor
from torch.nn import Module, Linear
from typing import TypedDict

from .encoders import FileEncoder
from .batching import Batch


class ModelCfg(TypedDict):
    depth:              int
    num_heads:          int
    dim:                int
    head_dim:           int


class Model(Module):
    def __init__(self, config: ModelCfg):
        super(Model, self).__init__()
        self.file_encoder = FileEncoder(
            num_layers=config['depth'],
            num_heads=config['num_heads'],
            head_dim=config['head_dim'],
            dim=config['dim'],
            dropout_rate=0.1
        )
        self.lemma_predictor = Linear(config['dim'], 1)

    def encode(self, batch: Batch) -> tuple[Tensor, Tensor]:
        return self.file_encoder.forward(
            scope_asts=batch.dense_scopes,
            scope_sort=batch.scope_sort,
            hole_asts=batch.dense_holes)

    def match(self, scope_reprs: Tensor, hole_reprs: Tensor, edge_index: Tensor) -> Tensor:
        source_index, target_index = edge_index
        sources = scope_reprs[source_index]
        targets = hole_reprs[target_index]
        return self.lemma_predictor.forward(sources * targets).squeeze(-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str, strict: bool = True) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location), strict=strict)
