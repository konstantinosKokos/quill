import pdb

from .data.reader import File, enum_references, Name
from .data.tokenization import tokenize_file
from .neural.model import Model
from .neural.batching import make_collator, device


class Wrapper:
    def __init__(self, cast_to: str):
        self.model = Model(num_layers=8, dim=128, max_db_index=20).to(cast_to)
        self.collator = make_collator(cast_to=device(cast_to))

    def forward(self, file: File[str], threshold: float = 0.5) -> list[list[Name]]:
        anonymized, backreferences = enum_references(file)
        tokenized = tokenize_file(anonymized)
        batch = self.collator([tokenized], 0)
        scope_reprs, goal_reprs = self.model.encode(batch)
        num_scopes, num_goals = scope_reprs.shape[1], goal_reprs.shape[1]
        lemma_predictions = self.model.predict_lemmas(scope_reprs=scope_reprs[:, :, 0],
                                                      goal_reprs=goal_reprs[:, :, 0],
                                                      edge_index=batch.edge_index)
        lemma_predictions = lemma_predictions.sigmoid().ge(threshold).view(num_goals, num_scopes).cpu().tolist()
        suggestions = [[idx for idx, value in enumerate(hole) if value] for hole in lemma_predictions]
        return [[backreferences[idx] for idx in hole] for hole in suggestions]
