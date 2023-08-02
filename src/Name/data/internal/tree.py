from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass

L1 = TypeVar('L1')
O1 = TypeVar('O1')
L2 = TypeVar('L2')
O2 = TypeVar('O2')


class TreeBase(ABC, Generic[L1, O1]):
    def __init__(self):
        raise ValueError("Can't instantiate abstract class TreeBase")

    @abstractmethod
    def depth(self) -> int: ...


@dataclass
class Binary(TreeBase[L1, O1]):
    op: O1
    left: TreeBase[L1, O1]
    right: TreeBase[L1, O1]

    def depth(self) -> int: return 1 + max(self.left.depth(), self.right.depth())


@dataclass
class Terminal(TreeBase[L1, O1]):
    content: L1

    def depth(self) -> int: return 0


def tree_zip(left: TreeBase[L1, O1], right: TreeBase[L2, O2]) -> TreeBase[tuple[L1, L2], tuple[O1, O2]]:
    match left, right:
        case Binary(o1, l1, r1), Binary(o2, l2, r2): return Binary((o1, o2), tree_zip(l1, l2), tree_zip(r1, r2))
        case Terminal(c1), Terminal(c2): return Terminal((c1, c2))
        case _: raise ValueError


def enumerate_nodes(tree: TreeBase[L1, O1]) -> TreeBase[tuple[L1, int], tuple[O1, int]]:
    def go(_tree: TreeBase[L1, O1], start: int) -> TreeBase[tuple[L1, int], tuple[O1, int]]:
        match _tree:
            case Binary(op, left, right): return Binary((op, start), go(left, 2 * start), go(right, 2 * start + 1))
            case Terminal(content): return Terminal((content, start))
            case _: raise ValueError
    return go(tree, 1)


def flatten(tree: TreeBase[L1, O1]) -> list[L1 | O1]:
    match tree:
        case Binary(op, left, right): return [op, *flatten(left), *flatten(right)]
        case Terminal(content): return [content]
        case _: raise ValueError
