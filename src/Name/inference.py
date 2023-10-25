import pdb

import torch

from .data.agda.reader import File
from .data.tokenization import tokenize_file
from .neural.model import Model, ModelCfg
from .neural.batching import Collator
from .neural.train import Logger

from torch_geometric.utils import to_dense_batch

import sys


class Inferer(Model):
    def __init__(self, model_config: ModelCfg, cast_to: str):
        super(Inferer, self).__init__(model_config)
        self.collator = Collator(pad_value=-1, allow_self_loops=False, device=cast_to)
        self.eval()

    def select_premises(self, file: File[str], threshold: float = 0.5) -> list[set[str]]:
        tokenized = tokenize_file(file)

        with torch.no_grad():
            batch = self.collator([tokenized])
            scope_reprs, hole_reprs = self.encode(batch)
            lemma_predictions = self.predict_lemmas(scope_reprs, hole_reprs, batch.edge_index)
            sparse = to_dense_batch(lemma_predictions, batch.edge_index[1], fill_value=-1e8)
            # pdb.set_trace()
            # # todo
            # # rounded = lemma_predictions.sigmoid().ge(threshold)
            # # return [{tokenized.backrefs[idx] for idx in hole} for hole]
    #
    # def batch_eval(self, files: list[File[str]], threshold: float = 0.5):
    #     logger = Logger(sys.stdout, './blah')
    #     sys.stdout = logger
    #
    #     for file in files:
    #         try:
    #             predictions = self.select_premises(file, threshold)
    #         except ValueError:
    #             continue
    #         logger = Logger(logger.stdout, f'./{file.name}.txt')
    #         sys.stdout = logger
    #
    #         print('*' * 32 + 'SCOPE' + '*' * 32)
    #         print('\n'.join(f'{entry.name}\n{"-" * 64}\n{entry.type}\n{entry.definition}\n' for entry in file.scope))
    #         print()
    #         print('*' * 32 + 'HOLES' + '*' * 32)
    #         for i, hole in enumerate(file.holes):
    #             print(f'Hole type: {hole.goal}')
    #             print(f'Predicted lemmas: {predictions[i]}')
    #             print(f'Definition was: {hole.term}')
    #             print(f'Gold lemmas: {hole.premises}')
    #             print()
    #         print()
