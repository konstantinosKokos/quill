from .reader import AgdaType, PiType, LamType, AppType, Reference, DeBruijn, Declaration, LitType, LevelType, SortType
from .tree import Binary, Terminal, TreeBase
from enum import Enum


class OpNames(Enum):
    App = 'App'
    Lam = 'Lam'
    Pi = 'Pi'


class DontCare(Enum):
    Sort = 'Sort'
    Level = 'Level'
    Lit = 'Lit'


AgdaTree = TreeBase[Reference | DeBruijn | DontCare, OpNames]


def strip_name(type_or_declaration: AgdaType | Declaration) -> AgdaType:
    return type_or_declaration.type if isinstance(type_or_declaration, Declaration) else type_or_declaration


def agda_to_tree(agda_type: AgdaType) -> AgdaTree:
    match agda_type:
        case PiType(argument, result):
            return Binary(OpNames.Pi, agda_to_tree(strip_name(argument)), agda_to_tree(result))
        case LamType(abstraction, body):
            raise NotImplementedError
            # return Binary(OpNames.Lam, ..., agda_to_tree(body))  # todo
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
