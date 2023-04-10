import pdb

from .reader import File
from .internal import AgdaTree, DontCare, DeBruijn, Reference, OpNames, agda_to_tree
from .tree import enumerate_nodes, flatten


TokenizedNode = tuple[int, int, int]
TokenizedTree = list[TokenizedNode]


def tokenize_node(node: tuple[Reference | DeBruijn | DontCare | OpNames, int]) -> TokenizedNode:
    content, position = node
    match content:
        case OpNames(): return 0, content.value, position
        case DontCare(): return 1, content.value, position
        case Reference(name): return 2, name, position
        case DeBruijn(index): return 3, index, position
        case _: raise ValueError


def detokenize_node(node: tuple[int, int]) -> Reference | DeBruijn | DontCare | OpNames:
    match node:
        case 0, value: return OpNames(value)
        case 1, value: return DontCare(value)
        case 2, value: return Reference(value)
        case 3, value: return DeBruijn(value)
        case _: raise ValueError


def tokenize_tree(agda_tree: AgdaTree) -> TokenizedTree:
    # given an agda tree, yields its nods in the form of (token_type, token_value, token_position) in BFT
    flat: list[tuple[Reference | DeBruijn | DontCare | OpNames, int]]
    flat = flatten(enumerate_nodes(agda_tree))
    return [tokenized for node in flat if (tokenized := tokenize_node(node))[:2] != (1, 3)]  # todo: ad-hoc filter


def detokenize_tree(nodes: TokenizedTree) -> AgdaTree:
    raise NotImplementedError


def tokenize_file(file: File[int]) -> tuple[list[TokenizedTree], list[tuple[list[TokenizedTree], list[int]]]]:
    scope: list[TokenizedTree]
    scope = [tokenize_tree(agda_to_tree(declaration.type)) for declaration in file.scope]
    samples: list[tuple[list[TokenizedTree], list[int]]]
    samples = [([tokenize_tree(agda_to_tree(declaration.type)) for declaration in sample.context],
                [ref.name for ref in sample.names_used])
               for sample in file.samples]
    return scope, samples


def detokenize_file(*args, **kwargs) -> File[int]:
    raise NotImplementedError
