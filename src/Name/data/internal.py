from .reader import AgdaType, PiType, LamType, AppType, Reference, DeBruijn, LitType, LevelType, SortType, strip_name
from .tree import Binary, Terminal, TreeBase
from enum import Enum, unique


@unique
class OpNames(Enum):
    App = 0
    Lam = 1
    Pi = 2


@unique
class DontCare(Enum):
    Sort = 0
    Level = 1
    Lit = 2
    Abs = 3


AgdaTree = TreeBase[Reference | DeBruijn | DontCare, OpNames]


def agda_to_tree(agda_type: AgdaType) -> AgdaTree:
    match agda_type:
        case PiType(argument, result):
            return Binary(OpNames.Pi, agda_to_tree(strip_name(argument)), agda_to_tree(result))
        case LamType(_, body):
            return Binary(OpNames.Lam, Terminal(DontCare.Abs), agda_to_tree(body))
        case AppType(head, argument):
            return Binary(OpNames.App, agda_to_tree(head), agda_to_tree(argument))
        case Reference(name):
            return Terminal(Reference(name))
        case DeBruijn(index):
            return Terminal(DeBruijn(index))
        case SortType(_):
            return Terminal(DontCare.Sort)
        case LevelType(_):
            return Terminal(DontCare.Level)
        case LitType(_):
            return Terminal(DontCare.Lit)
        case _:
            raise ValueError
