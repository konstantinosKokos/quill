import torch

from ..data.agda.reader import File
from ..data.tokenization import tokenize_file
from .model import Model, ModelCfg
from .batching import Collator
from .utils.ranking import rank_candidates
from torch import Tensor


class Inferer(Model):
    def __init__(self, model_config: ModelCfg, cast_to: str):
        super(Inferer, self).__init__(model_config)
        self.collator = Collator(pad_value=-1, allow_self_loops=False, device=cast_to)
        self.eval()
        self.to(cast_to)

    def select_premises(self, file: File[str]) -> list[list[str]]:
        tokenized = tokenize_file(file, merge_holes=False, unique_only=False)
        if file.num_holes == 0:
            return []

        with torch.no_grad():
            batch = self.collator([tokenized])
            scope_reprs, hole_reprs = self.encode(batch)
            pair_scores = self.match(scope_reprs, hole_reprs, batch.edge_index)
            ranked, numels = rank_candidates(pair_scores, batch.edge_index[1])
            return [[tokenized.backrefs[idx] for idx in perm[:valid]]
                    for perm, valid in zip(ranked.cpu().tolist(), numels.cpu().tolist())]

    def extract_reprs(self, file: File[str]) -> list[tuple[str, Tensor]]:
        tokenized = tokenize_file(file, merge_holes=False, unique_only=False)
        if file.num_holes == 0:
            return []

        with torch.no_grad():
            batch = self.collator([tokenized])
            scope_reprs, _ = self.encode(batch)
            return [(entry.name, tensor) for entry, tensor in zip(file.scope, scope_reprs) if not entry.is_import]
