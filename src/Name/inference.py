import torch

from .data.agda.reader import File
from .data.tokenization import tokenize_file
from .nn.model import Model, ModelCfg
from .nn.batching import Collator
from .nn.utils.ranking import rank_candidates


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


# from Name.inference import Inferer
# from scripts.train import model_cfg, dev_files
#
# inferer = Inferer(model_cfg, 'cuda')
# inferer.load('./scripts/model.pt', 'cuda')
# from Name.data.agda.reader import parse_file
# out = inferer.select_premises(f := parse_file(f'./data/stdlib/{dev_files[43]}.json'))