from .reader import File
from .internal import AgdaTree, DontCare, DeBruijn, Reference, OpNames, agda_to_tree
from .tree import enumerate_nodes, flatten


TokenizedNode = tuple[int, int, int, int]
TokenizedTree = list[TokenizedNode]
TokenizedTrees = list[TokenizedTree]
TokenizedSample = tuple[TokenizedTrees, list[tuple[TokenizedTree, list[int]]]]
TokenizedFile = TokenizedSample
SOS: TokenizedNode = (0, 0, 0, 0)


def tokenize_node(node: tuple[Reference | DeBruijn | DontCare | OpNames, int],
                  tree_index: int) -> TokenizedNode:
    content, position = node
    match content:
        case OpNames(): return 1, content.value, position, tree_index
        case DontCare(): return 2, content.value, position, tree_index
        case Reference(name): return 3, name, position, tree_index
        case DeBruijn(index): return 4, index, position, tree_index
        case _: raise ValueError


def detokenize_node(node: tuple[int, int]) -> Reference | DeBruijn | DontCare | OpNames:
    match node:
        case 1, value: return OpNames(value)
        case 2, value: return DontCare(value)
        case 3, value: return Reference(value)
        case 4, value: return DeBruijn(value)
        case _: raise ValueError


def tokenize_tree(agda_tree: AgdaTree, tree_index: int) -> TokenizedTree:
    # todo: deal with [sos] token
    # given an agda tree, yields its nods in the form of (token_type, token_value, token_pos, tree_pos) in BFT
    flat: list[tuple[Reference | DeBruijn | DontCare | OpNames, int]]
    flat = flatten(enumerate_nodes(agda_tree))
    return [SOS] + [tokenize_node((content, idx), tree_index) for content, idx in flat if content != DontCare.Abs]


def detokenize_tree(nodes: TokenizedTree) -> AgdaTree:
    raise NotImplementedError


def tokenize_file(file: File[int]) -> TokenizedFile:
    scope: list[TokenizedTree]
    scope = [tokenize_tree(agda_to_tree(declaration.type), i) for i, declaration in enumerate(file.scope)]
    goals: list[tuple[TokenizedTree, list[int]]]
    # todo: deal with positional index of goal type
    goals = [(tokenize_tree(agda_to_tree(hole.goal_type), -1), [ref.name for ref in hole.names_used])
             for hole in file.holes]
    return scope, goals


def detokenize_file(*args, **kwargs) -> File[int]:
    raise NotImplementedError
