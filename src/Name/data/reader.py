from __future__ import annotations

from json import load
from os import listdir, path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeVar, Any, Iterator, Optional
from typing_extensions import Self
from collections import defaultdict

Name = TypeVar('Name')
Other = TypeVar('Other')


@dataclass(unsafe_hash=True)
class File(Generic[Name]):
    name: str
    scope: list[Declaration[Name]]
    holes: list[Hole[Name]]

    def __repr__(self) -> str:
        return '\n'.join(f'{d}' for d in self.scope) + f'\n{"="*64}\n\n' + '\n'.join(f'{s}' for s in self.holes)


@dataclass(unsafe_hash=True)
class Hole(Generic[Name]):
    type: AgdaTerm[Name]
    definition: AgdaTerm[Name]
    premises: list[Reference[Name]]  # n.b. scope reference only

    def __repr__(self) -> str:
        return f'\t{self.definition} : {self.type}'


class AgdaTerm(ABC, Generic[Name]):
    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def substitute(self, names: dict[Name, Other]) -> Self[Other]: ...


@dataclass(unsafe_hash=True)
class Declaration(AgdaTerm[Name]):
    name: Name
    type: AgdaTerm[Name]
    definition: AgdaTerm[Name]

    def __repr__(self) -> str: return f'{self.name} :: {self.type}\n{self.name} = {self.definition}'

    def substitute(self, names: dict[Name, Other]) -> Declaration[Other]:
        return Declaration(names[self.name], self.type.substitute(names), self.definition.substitute(names))


def strip_name(type_or_declaration: AgdaTerm | Declaration) -> AgdaTerm:
    return type_or_declaration.type if isinstance(type_or_declaration, Declaration) else type_or_declaration


@dataclass(unsafe_hash=True)
class PiTerm(AgdaTerm[Name]):
    domain: AgdaTerm[Name]
    codomain: AgdaTerm[Name]
    name: Optional[Name]

    def __repr__(self) -> str:
        if self.name is not None:
            dom_repr = f'({self.name} : {self.domain})'
        elif isinstance(self.domain, PiTerm):
            dom_repr = f'({self.domain})'
        else:
            dom_repr = f'{self.domain}'
        return f'{dom_repr} -> {self.codomain}'

    def substitute(self, names: dict[Name, Other]) -> PiTerm[Other]:
        return PiTerm(domain=self.domain.substitute(names),
                      codomain=self.codomain.substitute(names),
                      name=None if self.name is None else names[self.name])


@dataclass(unsafe_hash=True)
class LamTerm(AgdaTerm[Name]):
    abstraction: Any
    body: AgdaTerm[Name]

    def __repr__(self) -> str: return f'λ{self.abstraction}.{self.body}'

    def substitute(self, names: dict[Name, Other]) -> LamTerm[Other]:
        return LamTerm(self.abstraction, self.body.substitute(names))


@dataclass(unsafe_hash=True)
class AppTerm(AgdaTerm[Name]):
    head: Reference[Name] | DeBruijn
    argument: AgdaTerm[Name]

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, AppTerm) else f'{self.argument}'
        return f'{self.head} {arg_repr}'

    def substitute(self, names: dict[Name, Other]) -> AppTerm[Other]:
        return AppTerm(self.head.substitute(names), self.argument.substitute(names))


@dataclass(unsafe_hash=True)
class Reference(AgdaTerm[Name]):
    name: Name

    def __repr__(self) -> str: return f'{self.name}'

    def substitute(self, names: dict[Name, Other]) -> Reference[Other]:
        return Reference(names[self.name])


@dataclass(unsafe_hash=True)
class DeBruijn(AgdaTerm[Name]):
    index: int

    def __repr__(self) -> str: return f'@{self.index}'

    def substitute(self, names: dict[Name, Other]) -> DeBruijn[Other]:
        return self


@dataclass(unsafe_hash=True)
class LitTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LitTerm[Other]:
        return self


@dataclass(unsafe_hash=True)
class SortTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> SortTerm[Other]:
        return self


@dataclass(unsafe_hash=True)
class LevelTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LevelTerm[Other]:
        return self


@dataclass(unsafe_hash=True)
class ADTTerm(AgdaTerm[Name]):
    variants: tuple[AgdaTerm[Name], ...]

    def __repr__(self) -> str: return ' | '.join(f'{var}' for var in self.variants)

    def substitute(self, names: dict[Name, Other]) -> ADTTerm[Other]:
        return ADTTerm(tuple(variant.substitute(names) for variant in self.variants))


@dataclass(unsafe_hash=True)
class Constructor(AgdaTerm[Name]):
    reference: Reference
    variant: int

    def __repr__(self) -> str: return f'{self.reference}[{self.variant}]'

    def substitute(self, names: dict[Name, Other]) -> Constructor[Other]:
        return Constructor(self.reference.substitute(names), self.variant)


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
                premises=[Reference(p) for p in hole_json['premises']])


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
            return SortTerm(content=term_json['sort'])
        case 'Literal':
            return LitTerm(content=term_json['literal'])
        case 'Level':
            return LevelTerm(content=term_json['level'])
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
                         premises=[premise.substitute(name_to_index) for premise in hole.premises])
                    for hole in file.holes]),
        index_to_name)
