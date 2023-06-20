import torch
from torch import Tensor
from .encoders import ScopeEncoder
from .batching import Batch
from .utils.modules import focal_loss
from torch.nn import Module, Bilinear
from torch.nn.functional import cross_entropy


class Model(Module):
    def __init__(self, num_layers: int, dim: int, max_db_index: int):
        super(Model, self).__init__()
        self.scope_encoder = ScopeEncoder(
            num_layers=num_layers,
            num_ops=3,
            num_leaves=4,
            max_db_index=max_db_index,
            dim=dim)
        self.lemma_predictor = Bilinear(dim, dim, 1)

    def predict_lemmas(self, scope_reprs: Tensor, goal_reprs: Tensor, edge_index: Tensor) -> Tensor:
        source_index, target_index = edge_index
        sources = scope_reprs.flatten(0, 1)[source_index]
        targets = goal_reprs.flatten(0, 1)[target_index]
        return self.lemma_predictor(sources, targets).squeeze(-1)

    def predict_masks(self, token_reprs: Tensor, lm_mask: Tensor, batch_pts: Tensor, tree_mask: Tensor) -> Tensor:
        mask_reprs = token_reprs[lm_mask]
        type_reprs = token_reprs[:, :, 0]
        candidates = torch.index_select(type_reprs, 0, batch_pts)
        candidate_mask = torch.index_select(tree_mask, 0, batch_pts)
        lm_preds = (mask_reprs.unsqueeze(-2) * candidates).sum(-1)
        lm_preds[~candidate_mask] = -1e10
        return lm_preds

    def encode(self, batch: Batch) -> tuple[Tensor, Tensor]:
        return self.scope_encoder.forward(batch)

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        scope_token_reprs, goal_token_reprs = self.encode(batch)
        lemma_predictions = self.predict_lemmas(
            scope_reprs=scope_token_reprs[:, :, 0],
            goal_reprs=goal_token_reprs[:, :, 0],
            edge_index=batch.edge_index)
        lm_predictions = self.predict_masks(
            token_reprs=scope_token_reprs,
            lm_mask=batch.lm_mask,
            batch_pts=batch.batch_pts,
            tree_mask=batch.scope_tree_mask)
        return lemma_predictions, lm_predictions

    def compute_losses(self, batch: Batch) -> tuple[list[bool], list[bool], tuple[int, int], Tensor, Tensor]:
        lemma_preds, lm_preds = self.forward(batch)

        lemma_loss = focal_loss(lemma_preds, batch.gold_lemmas, gamma=2.)
        lm_loss = cross_entropy(lm_preds, batch.masked_values, reduction='sum')

        lm_hits = lm_preds.argmax(dim=-1).eq(batch.masked_values).cpu()

        return (lemma_preds.sigmoid().round().cpu().bool().tolist(),
                batch.gold_lemmas.cpu().tolist(),
                (sum(lm_hits), lm_hits.numel()),
                lemma_loss,
                lm_loss)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = 'cpu'):
        self.load_state_dict(torch.load(path, map_location=map_location), strict=True)
