from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch import Tensor, empty, long, device


def pad_sequence(
        sequences: list[Tensor],
        padding_value: int,
        default_size: tuple[int, ...],
        default_device: device | str) -> Tensor:
    if sequences:
        return _pad_sequence(sequences, padding_value=padding_value, batch_first=True)
    return empty(default_size, dtype=long, device=default_device)