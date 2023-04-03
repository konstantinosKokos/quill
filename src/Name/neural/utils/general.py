from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def pad_sequence(xs: list[Tensor], padding_value: float) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)
