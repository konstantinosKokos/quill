from .tree import Binary, Terminal, TreeBase
from ..agda.syntax import (AgdaTerm, PiTerm, LamTerm, AppTerm, Reference, DeBruijn,
                           LitTerm, LevelTerm, SortTerm, ADTTerm, Constructor)
from enum import Enum, unique


@unique
class BinaryOps(Enum):
    App = 0
    Lam = 1
    Pi = 2
    ADT = 3
    Constructor = 4


@unique
class UnaryOps(Enum):
    Reference = 0
    deBruijn = 1
    Variant = 2


@unique
class NullaryOps(Enum):
    Sort = 0
    Level = 1
    Lit = 2
    Abs = 3


AgdaTree = TreeBase[tuple[UnaryOps, int] | NullaryOps, BinaryOps]


def term_to_tree(agda_term: AgdaTerm) -> AgdaTree:
    match agda_term:
        case PiTerm(domain, codomain, _):
            return Binary(op=BinaryOps.Pi, left=term_to_tree(domain), right=term_to_tree(codomain))
        case AppTerm(head, argument):
            return Binary(op=BinaryOps.App, left=term_to_tree(head), right=term_to_tree(argument))
        case LamTerm(_, body):
            return Binary(op=BinaryOps.Lam, left=Terminal(NullaryOps.Abs), right=term_to_tree(body))
        case Reference(name):
            return Terminal((UnaryOps.Reference, name))
        case DeBruijn(index):
            return Terminal((UnaryOps.deBruijn, index))
        case SortTerm(_):
            return Terminal(NullaryOps.Sort)
        case LevelTerm(_):
            return Terminal(NullaryOps.Level)
        case LitTerm(_):
            return Terminal(NullaryOps.Lit)
        case ADTTerm(variants):
            # todo: singleton / empty data types?
            fst, *rest = variants
            return Binary(op=BinaryOps.ADT,
                          left=term_to_tree(fst),
                          right=term_to_tree(rest[0]) if len(rest) == 1 else term_to_tree(ADTTerm(rest)))
        case Constructor(reference, variant):
            return Binary(op=BinaryOps.Constructor,
                          left=term_to_tree(reference),
                          right=Terminal((UnaryOps.Variant, variant)))
    raise ValueError


def tree_to_term(tree: AgdaTree) -> AgdaTerm:
    raise NotImplementedError
