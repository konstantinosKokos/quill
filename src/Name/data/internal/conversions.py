from ..agda.syntax import (AgdaTerm, PiTerm, LamTerm, AppTerm, Reference, DeBruijn, LitTerm,
                           SortTerm, LevelTerm, UnsolvedMeta,
                           AgdaDefinition, ADT, Constructor, Record, Function, FunctionClause,
                           Postulate, Primitive,
                           Hole)
from .tree import Binary, Terminal, TreeBase
from enum import Enum, unique
from typing import NoReturn


@unique
class BinaryOp(Enum):
    PiSimple = 0
    PiDependent = 1
    Lambda = 2
    Application = 3


@unique
class NullaryOps(Enum):
    Sort = 0
    Level = 1
    Literal = 2
    Abs = 3


AgdaLeaf = NullaryOps | Reference | DeBruijn
AgdaOp = BinaryOp
AgdaTree = TreeBase[tuple[int, AgdaLeaf], tuple[int, AgdaOp]]


def term_to_ast(term: AgdaTerm, n: int = 1, bindings: tuple[int, ...] = ()) -> AgdaTree:
    match term:
        case PiTerm(domain, codomain, None):
            return Binary(
                op=(n, BinaryOp.PiSimple),
                left=term_to_ast(domain, 2 * n, bindings),
                right=term_to_ast(codomain, 2 * n + 1, bindings))
        case PiTerm(domain, codomain, _):
            return Binary(
                op=(n, BinaryOp.PiDependent),
                left=term_to_ast(domain, 2 * n, bindings),
                right=term_to_ast(codomain, 2 * n + 1, (2 * n, *bindings)))
        case LamTerm(_, body):
            return Binary(
                op=(n, BinaryOp.Lambda),
                left=Terminal((2 * n, NullaryOps.Abs)),
                right=term_to_ast(body, 2 * n + 1, (2 * n, *bindings)))
        case AppTerm(head, argument):
            return Binary(
                op=(n, BinaryOp.Application),
                left=term_to_ast(head, 2 * n, bindings),
                right=term_to_ast(argument, 2 * n + 1, bindings))
        case Reference(_):
            return Terminal((n, term))
        case DeBruijn(index):
            return Terminal((n, DeBruijn(bindings[index])))
        case SortTerm(_):
            return Terminal((n, NullaryOps.Sort))
        case LitTerm(_):
            return Terminal((n, NullaryOps.Literal))
        case LevelTerm(_):
            return Terminal((n, NullaryOps.Level))
        case _:
            raise ValueError


def definition_to_ast(definition: AgdaDefinition) -> NoReturn:
    match definition:
        case ADT(variants):
            raise NotImplementedError
        case Constructor(reference, variant):
            raise NotImplementedError
        case Record(fields, telescope):
            raise NotImplementedError
        case Function(clauses):
            raise NotImplementedError
        case Postulate():
            raise NotImplementedError
        case Primitive():
            raise NotImplementedError
        case _:
            raise ValueError


def clause_to_ast(clause: FunctionClause) -> NoReturn:
    raise NotImplementedError

