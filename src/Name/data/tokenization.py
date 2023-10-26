from .internal.conversions import AgdaLeaf, AgdaOp, AgdaTree, NullaryOps, BinaryOp, Reference, DeBruijn, term_to_ast
from .internal.tree import flatten
from .agda.operations import enum_references, merge_contexts, top_sort_entries
from .agda.syntax import File

from typing import NoReturn, NamedTuple


TokenizedNode = tuple[int, int, int]
TokenizedAST = list[TokenizedNode]


def tokenize_node(node: AgdaOp | AgdaLeaf,
                  node_idx: int) -> TokenizedNode:
    match node:
        case BinaryOp(): return 1, node.value, node_idx
        case NullaryOps(): return 2, node.value, node_idx
        case Reference(name): return 3, name, node_idx
        case DeBruijn(index): return 4, index, node_idx
        case _: raise ValueError(node)


def tokenize_ast(ast: AgdaTree) -> TokenizedAST:
    flat = flatten(ast)
    return [(0, 0, 0),
            *[tokenize_node(content, idx) for idx, content in flat if content != NullaryOps.Abs]]


def detokenize_node(node: tuple[int, int]) -> NoReturn:
    raise NotImplementedError


def detokenize_ast(nodes: TokenizedAST) -> NoReturn:
    raise NotImplementedError


class TokenizedFile(NamedTuple):
    file:               File[str]
    backrefs:           dict[int, str]
    entry_sort:         list[int]
    scope_asts:         list[TokenizedAST]
    hole_to_scope:      list[int]
    hole_asts:          list[TokenizedAST]
    premises:           list[list[int]]


def tokenize_file(original: File[str]) -> TokenizedFile:
    merged = merge_contexts(original, merge_holes=True, unique_only=True)
    anonymous, backrefs = enum_references(merged)
    entry_sort = top_sort_entries(anonymous)
    zipped_scopes = tuple(zip(*[
        (i, tokenize_ast(term_to_ast(entry.type, 1, ()))) for i, entry in enumerate(anonymous.scope)]))
    zipped_holes = tuple(zip(*[
        (i, tokenize_ast(term_to_ast(hole.goal, 1, ())), [n for lemma in hole.premises if (n := lemma.name) != 1])
        for i, entry in enumerate(anonymous.scope) for hole in entry.holes]))
    scope_positions, scope_asts = zipped_scopes or ([], [])
    hole_to_scope, hole_asts, premises = zipped_holes or ([], [], [])
    return TokenizedFile(
        file=original,
        backrefs=backrefs,
        entry_sort=[entry_sort[i] for i in range(len(entry_sort))],
        scope_asts=scope_asts,
        hole_to_scope=hole_to_scope,
        hole_asts=hole_asts,
        premises=premises
    )


def detokenize_file() -> NoReturn:
    raise NotImplementedError
