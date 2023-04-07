from .reader import (AgdaType, PiType, LamType, AppType, Reference, DeBruijn, Declaration, LitType, LevelType, SortType,
                     File, Hole)
from .tree import Binary, Terminal, TreeBase
from collections import defaultdict
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
            return Binary(OpNames.Lam, abstraction, agda_to_tree(body))
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


def enum_references(file: File[str]) -> File[int]:
    name_to_index = defaultdict(lambda: -1,
                               {declaration.name: idx for idx, declaration in enumerate(file.scope)})
    return File(name=file.name,
                scope=[Declaration(name=index,
                                   type=declaration.type.substitute(name_to_index))
                       for index, declaration in enumerate(file.scope)],
                samples=[Hole(context=[Declaration(name=declaration.name,
                                                   type=declaration.type.substitute(name_to_index))
                                       for declaration in hole.context],
                              goal_type=hole.goal_type.substitute(name_to_index),
                              goal_term=hole.goal_term.substitute(name_to_index),
                              names_used=[name.substitute(name_to_index) for name in hole.names_used])
                         for hole in file.samples])

