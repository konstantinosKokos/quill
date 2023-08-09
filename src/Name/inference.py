from .data.agda.reader import File, enum_references
from .data.tokenization import tokenize_file
from .neural.model import Model, ModelCfg
from .neural.batching import Collator, device


class Inferer(Model):
    def __init__(self, model_config: ModelCfg, cast_to: str):
        super(Inferer, self).__init__(model_config)
        self.collator = Collator(pad_value=-1, cast_to=device(cast_to))

    def select_premises(self, file: File[str], threshold: float = 0.5) -> list[set[str]]:
        anonymized, backreferences = enum_references(file)
        tokenized = tokenize_file(anonymized)
        batch = self.collator([tokenized])
        scope_reprs, _, _, hole_reprs = self.model.encode(batch)
        num_scopes, num_goals = scope_reprs.shape[1], hole_reprs.shape[1]
        lemma_predictions = self.model.predict_lemmas(scope_reprs=scope_reprs,
                                                      goal_reprs=hole_reprs[:, :, 0],
                                                      edge_index=batch.edge_index)
        lemma_predictions = lemma_predictions.sigmoid().ge(threshold).view(num_goals, num_scopes).cpu().tolist()
        suggestions = [[idx for idx, value in enumerate(hole) if value] for hole in lemma_predictions]
        return [{backreferences[idx] for idx in hole} for hole in suggestions]
