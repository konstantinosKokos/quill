from .syntax import (AgdaTerm, AppTerm, PiTerm, LamTerm, LitTerm, SortTerm,
                     LevelTerm, DeBruijn, Reference, UnsolvedMeta,
                     AgdaDefinition, ADT, Constructor, Record, Function,
                     FunctionClause, Postulate, Primitive,
                     File, ScopeEntry, Hole, NamedType)
from json import load
from os import listdir, path
from functools import reduce
from typing import Iterator, Callable, TypeVar

T = TypeVar('T')
DEBUG = False


def debug(wrapped: Callable[[dict], T]) -> Callable[[dict], T]:
    def wrapper(json: dict) -> T:
        print(json)
        return wrapped(json)
    return wrapper if DEBUG else wrapped


def parse_dir(directory: str, strict: bool, validate: bool = True) -> Iterator[File[str]]:
    for file in listdir(directory):
        print(f'Parsing {file}')
        try:
            yield parse_file(path.join(directory, file), validate)
        except AssertionError as e:
            if strict:
                raise e
            print(f'\tFailed: {e}.')
            continue


def parse_file(filepath: str, validate: bool) -> File[str]:
    with open(filepath, 'r') as f:
        return parse_data(load(f), validate=validate)


def parse_data(json: dict, validate: bool) -> File[str]:
    return File(
        name=json['name'],
        scope=[parse_scope_entry(d, True) for d in json['scope-global']] +
              [parse_scope_entry(d, False) for d in json['scope-local']],
        validate=validate)


def parse_scope_entry(json: dict, is_import: bool) -> ScopeEntry[str]:
    return ScopeEntry(
        name=json['name'],
        type=parse_term(json['type']),
        definition=parse_definition(json['definition']),
        holes=[] if is_import else [parse_hole(hole) for hole in json['holes']],
        is_import=is_import)


@debug
def parse_hole(json: dict) -> Hole[str]:
    return Hole(
        context=tuple(parse_term(t) for t in json['ctx']['telescope']),
        goal=parse_term(json['goal']),
        term=parse_term(json['term']),
        premises=tuple(Reference(p) for p in json['premises']))


@debug
def parse_named_type(json: dict) -> NamedType:
    return NamedType(name=json['name'], type=parse_term(json))


@debug
def parse_definition(json: dict) -> AgdaDefinition[str]:
    match json['tag']:
        case 'ADT':
            return ADT(variants=tuple(parse_term(t) for t in json['variants']))
        case 'Constructor':
            return Constructor(reference=json['reference'], variant=json['variant'])
        case 'Record':
            return Record(
                fields=tuple(parse_term(f) for f in json['fields']),
                telescope=tuple(parse_term(t) for t in json['telescope']))
        case 'Function':
            return Function(clauses=tuple(parse_clause(c) for c in json['clauses']))
        case 'Postulate':
            return Postulate()
        case 'Primitive':
            return Primitive()
        case _:
            raise ValueError(f'Unknown tag {json["tag"]}')


@debug
def parse_clause(json: dict) -> FunctionClause:
    return FunctionClause(
        telescope=tuple(parse_named_type(nt) for nt in json['telescope']),
        patterns=tuple(parse_term(t) for t in json['patterns']),
        body=parse_term(json['body']) if 'body' in json.keys() else None)


@debug
def parse_term(json: dict) -> AgdaTerm[str]:
    match json['tag']:
        case 'Pi':
            return PiTerm(domain=parse_term(json['domain']),
                          codomain=parse_term(json['codomain']),
                          name=None if (name := json['name']) == '_' else name)
        case 'Application':
            return reduce(
                AppTerm,
                [parse_term(a) for a in json['arguments']],
                parse_term(json['head']))  # type: ignore
        case 'Lambda':
            return LamTerm(body=parse_term(json['body']), abstraction=json['abstraction'])
        case 'Sort':
            return SortTerm(content=json['sort'].replace(' ', '␣'))
        case 'Literal':
            return LitTerm(content=json['literal'].replace(' ', '␣'))
        case 'Level':
            return LevelTerm(content=json['level'].replace(' ', '␣'))
        case 'ScopeReference':
            return Reference(json['name'])
        case 'DeBruijn':
            return DeBruijn(json['index'])
        case 'UnsolvedMetavariable':
            return UnsolvedMeta()
        case _:
            raise ValueError(f'Unknown tag {json["tag"]}')
