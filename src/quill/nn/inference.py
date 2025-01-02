import torch

from ..data.agda.reader import File
from ..data.tokenization import tokenize_file
from .model import Model, ModelCfg
from .batching import Collator
from .utils.ranking import rank_candidates


class Inferer(Model):
    def __init__(self, model_config: ModelCfg, cast_to: str):
        super(Inferer, self).__init__(model_config)
        self.collator = Collator(pad_value=-1, allow_self_loops=False, device=cast_to)
        self.eval()
        self.to(cast_to)
        self.cache: dict[tuple[str, str], torch.Tensor] = dict()

    @torch.no_grad()
    def select_premises(self, file: File[str], with_cache: bool = False) -> list[list[str]]:
        tokenized = tokenize_file(file, merge_holes=False, unique_only=False)
        if file.num_holes == 0:
            return []

        with torch.no_grad():
            batch = self.collator([tokenized])
            (scope_reprs, hole_reprs), edge_index, backrefs = self.encode(batch), batch.edge_index, tokenized.backrefs
            if len(self.cache):
                (scope_reprs, hole_reprs, edge_index, backrefs) = self.extend_batch(
                    scope_reprs,
                    hole_reprs,
                    edge_index,
                    backrefs
                )
            pair_scores = self.match(scope_reprs, hole_reprs, edge_index)
            ranked, numels = rank_candidates(pair_scores, edge_index[1])
            return [[backrefs[idx] for idx in perm[:valid]]
                    for perm, valid in zip(ranked.cpu().tolist(), numels.cpu().tolist())]

    @torch.no_grad()
    def precompute(self, files: list[File[str]]) -> None:
        tokenizer = lambda f: tokenize_file(f, merge_holes=False, unique_only=False)
        for tokenized in map(tokenizer, files):
            if len(tokenized.scope_asts):
                batch = self.collator([tokenized])
                scope_reprs = self.encode_scope(batch)
                for i, lemma in tokenized.backrefs.items():
                    self.cache[(tokenized.file.name, lemma)] = scope_reprs[i].detach()

    def extend_batch(
            self,
            scope_reprs: torch.Tensor,
            hole_reprs: torch.Tensor,
            edge_index: torch.Tensor,
            backrefs: dict[int, str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, str]]:
        cache = torch.stack(list(self.cache.values())).to(scope_reprs.device)
        scope_size, cache_size = scope_reprs.size(0), cache.size(0)
        scope_reprs = torch.cat([scope_reprs, cache], dim=0)
        cache_index = torch.cartesian_prod(
            torch.arange(cache_size) + scope_size,
            torch.arange(hole_reprs.size(0))
        ).t().to(scope_reprs.device)
        edge_index = torch.cat((edge_index, cache_index), dim=1)
        backrefs = backrefs | {idx + scope_size: '.'.join(name) for idx, name in enumerate(self.cache.keys())}
        return scope_reprs, hole_reprs, edge_index, backrefs
