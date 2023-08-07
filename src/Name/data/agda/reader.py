from .syntax import (AgdaTerm, AppTerm, PiTerm, LamTerm, LitTerm, SortTerm, LevelTerm,
                     ADTTerm, Constructor, File, Declaration, DeBruijn, Reference, Hole)
from json import load
from os import listdir, path
from functools import reduce
from typing import Iterator
from collections import defaultdict


def parse_dir(directory: str, must_contain: str | None = None) -> Iterator[File[str]]:
    for file in listdir(directory):
        if (must_contain is None or must_contain in file) and file.endswith('.json'):
            print(f'Parsing {file}')
            yield parse_file(path.join(directory, file))


def parse_file(filepath: str) -> File[str]:
    with open(filepath, 'r') as f:
        return parse_data(load(f))


def parse_data(data_json: dict) -> File[str]:
    return File(name=data_json['name'],
                scope=[parse_declaration(d) for d in data_json['scope']['entries']],
                holes=[parse_hole(h) for h in data_json['holes']])


def parse_declaration(dec_json: dict) -> Declaration[str]:
    return Declaration(name=dec_json['name'],
                       type=parse_term(dec_json['type']['term']),
                       definition=parse_term(dec_json['definition']['term']))


def parse_hole(hole_json: dict) -> Hole[str]:
    return Hole(type=parse_term(hole_json['type']['term']),
                definition=parse_term(hole_json['definition']['term']),
                lemmas=[Reference(p) for p in hole_json['premises']])


def parse_term(term_json: dict) -> AgdaTerm[str]:
    match term_json['tag']:
        case 'Pi':
            return PiTerm(domain=parse_term(term_json['domain']),
                          codomain=parse_term(term_json['codomain']),
                          name=None if (name := term_json['name']) == '_' else name)
        case 'Application':
            return reduce(AppTerm,
                          [parse_term(a) for a in term_json['arguments']],
                          parse_term(term_json['head']))  # type: ignore
        case 'Lambda':
            return LamTerm(body=parse_term(term_json['body']), abstraction=term_json['abstraction'])
        case 'Sort':
            return SortTerm(content=term_json['sort'].replace(' ', '_'))
        case 'Literal':
            return LitTerm(content=term_json['literal'].replace(' ', '_'))
        case 'Level':
            return LevelTerm(content=term_json['level'].replace(' ', '_'))
        case 'ADT':
            return ADTTerm(tuple(parse_term(v) for v in term_json['variants']))
        case 'Constructor':
            return Constructor(reference=Reference(term_json['reference']), variant=int(term_json['variant']))
        case 'ScopeReference':
            return Reference(term_json['name'])
        case 'deBruijn':
            return DeBruijn(term_json['index'])
        case _:
            raise ValueError(f'Unknown tag f{term_json["tag"]}')


def enum_references(file: File[str]) -> tuple[File[int], dict[int, str]]:
    name_to_index = defaultdict(lambda: -1, {declaration.name: idx for idx, declaration in enumerate(file.scope)})
    index_to_name = {v: k for k, v in name_to_index.items()}
    return (
        File(name=file.name,
             scope=[Declaration(name=name_to_index[declaration.name],
                                type=declaration.type.substitute(name_to_index),
                                definition=declaration.definition.substitute(name_to_index))
                    for declaration in file.scope],
             holes=[Hole(type=hole.type.substitute(name_to_index),
                         definition=hole.definition.substitute(name_to_index),
                         lemmas=[premise.substitute(name_to_index) for premise in hole.lemmas])
                    for hole in file.holes]),
        index_to_name)
