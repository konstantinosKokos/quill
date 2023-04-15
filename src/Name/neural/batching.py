import torch
from torch import Tensor, device
from ..data.tokenization import TokenizedSample
from .utils import pad_to_length


def make_collator(cast_to: device = device('cpu'),
                  goal_id: int = -1,
                  padding: tuple[int, int, int, int] = (0, 0, 0, 0)):
    def collator(samples: list[TokenizedSample]) -> tuple[Tensor, Tensor, Tensor]:
        num_scopes = len(samples)
        num_trees = max(len(scope) + len(holes) for scope, holes in samples)
        all_trees = [scope + [goal_type for goal_type, _ in holes] for scope, holes in samples]

        num_tokens = max(max(len(tree) for tree in batch) for batch in all_trees)
        padded_trees = [pad_to_length(batch, num_tokens, padding) for batch in all_trees]
        padded_batches = pad_to_length(padded_trees, num_trees, [padding] * num_tokens)

        dense_batch = torch.tensor(padded_batches)
        token_padding_mask = (dense_batch != torch.tensor(padding)).all(dim=-1)
        tree_padding_mask = token_padding_mask.all(dim=-1).unsqueeze(-2).expand(-1, num_trees, -1)
        goal_mask = (dense_batch[:, :, :, -1] != goal_id).all(-1)
        diag_mask = torch.eye(num_trees, dtype=torch.bool).unsqueeze(0).expand(num_scopes, -1, -1)
        token_attention_mask = token_padding_mask.flatten(0, 1).unsqueeze(-2).expand(-1, num_tokens, -1)
        tree_attention_mask = tree_padding_mask & (goal_mask.unsqueeze(-1) | diag_mask)
        return (dense_batch.permute(-1, 0, 1, 2).to(cast_to),
                token_attention_mask.to(cast_to),
                tree_attention_mask.to(cast_to))
    return collator

