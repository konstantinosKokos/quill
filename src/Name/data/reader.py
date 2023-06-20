from __future__ import annotations

from json import load
from os import listdir, path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeVar, Any, Iterator
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
    goal_type: AgdaType[Name]
    goal_term: AgdaType[Name]
    names_used: list[Reference[Name]]  # n.b. scope reference only

    def __repr__(self) -> str:
        return f'\t{self.goal_term} : {self.goal_type}'


class AgdaType(ABC, Generic[Name]):
    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def substitute(self, names: dict[Name, Other]) -> Self[Other]: ...


@dataclass(unsafe_hash=True)
class Declaration(AgdaType[Name]):
    name: Name
    type: AgdaType[Name]

    def __repr__(self) -> str: return f'{self.name} :: {self.type}'

    def substitute(self, names: dict[Name, Other]) -> Declaration[Other]:
        return Declaration(names[self.name], self.type.substitute(names))


def strip_name(type_or_declaration: AgdaType | Declaration) -> AgdaType:
    return type_or_declaration.type if isinstance(type_or_declaration, Declaration) else type_or_declaration


Telescope = tuple[Declaration, ...]


@dataclass(unsafe_hash=True)
class PiType(AgdaType[Name]):
    argument: AgdaType[Name] | Declaration[Name]
    result: AgdaType[Name]

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, (Declaration, PiType)) else f'{self.argument}'
        return f'{arg_repr} -> {self.result}'

    def substitute(self, names: dict[Name, Other]) -> PiType[Other]:
        if isinstance(self.argument, Declaration):
            argument = Declaration(self.argument.name, self.argument.type.substitute(names))
        else:
            argument = self.argument.substitute(names)
        return PiType(argument, self.result.substitute(names))


@dataclass(unsafe_hash=True)
class LamType(AgdaType[Name]):
    abstraction: Any
    body: AgdaType[Name]

    def __repr__(self) -> str: return f'λ{self.abstraction}.{self.body}'

    def substitute(self, names: dict[Name, Other]) -> LamType[Other]:
        return LamType(self.abstraction, self.body.substitute(names))


@dataclass(unsafe_hash=True)
class AppType(AgdaType[Name]):
    head: Reference[Name] | DeBruijn
    argument: AgdaType[Name]

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, AppType) else f'{self.argument}'
        return f'{self.head} {arg_repr}'

    def substitute(self, names: dict[Name, Other]) -> AppType[Other]:
        return AppType(self.head.substitute(names), self.argument.substitute(names))


@dataclass(unsafe_hash=True)
class Reference(AgdaType[Name]):
    name: Name

    def __repr__(self) -> str: return f'{self.name}'

    def substitute(self, names: dict[Name, Other]) -> Reference[Other]:
        return Reference(names[self.name])


@dataclass(unsafe_hash=True)
class DeBruijn(AgdaType[Name]):
    index: int

    def __repr__(self) -> str: return f'@{self.index}'

    def substitute(self, names: dict[Name, Other]) -> DeBruijn[Other]:
        return self


@dataclass(unsafe_hash=True)
class LitType(AgdaType[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LitType[Other]:
        return self


@dataclass(unsafe_hash=True)
class SortType(AgdaType[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> SortType[Other]:
        return self


@dataclass(unsafe_hash=True)
class LevelType(AgdaType[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LevelType[Other]:
        return self


def parse_dir(directory: str, must_contain: str | None = None, version: str = 'original') -> Iterator[File[str]]:
    for file in listdir(directory):
        if (must_contain is None or must_contain in file) and file.endswith('.json'):
            print(f'Parsing {file}')
            yield parse_file(path.join(directory, file), version)


def parse_file(filepath: str, version: str) -> File[str]:
    with open(filepath, 'r') as f:
        return parse_data(load(f), version)


def parse_data(data_json: dict, version: str) -> File[str]:
    return File(name=data_json['scope']['name'],
                scope=[parse_declaration(d, version) for d in data_json['scope']['item']],
                holes=[parse_holes(s, version) for s in data_json['samples']])


def parse_holes(hole_json: dict, version: str) -> Hole[str]:
    context_json = hole_json['ctx']['thing']
    goal_type_json = hole_json['goal']
    goal_term_json = hole_json['term']
    goal_names_used = hole_json['namesUsed']
    context = [Declaration(name=c['name'], type=parse_type(c['item'], version)) for c in context_json]

    return Hole(
        goal_type=reduce(lambda result, argument: PiType(argument, result),
                         reversed(context),
                         parse_type(goal_type_json['thing'], version)),  # type: ignore
        goal_term=parse_type(goal_term_json['thing']['original'], version),
        names_used=[Reference(name) for name in goal_names_used])


def parse_declaration(dec_json: dict, version: str) -> Declaration[str]:
    return Declaration(name=dec_json['name'], type=parse_type(dec_json['item']['thing'], version))


def parse_type(type_json: dict, which: str) -> AgdaType[str]:
    def go(_type_json: dict) -> AgdaType[str]: return parse_type(_type_json, which)

    if which in type_json.keys():
        if (tmp := type_json[which]) is not None:
            type_json = tmp
        else:
            type_json = type_json['original']

    match type_json['tag']:
        case 'Pi':
            left, right = type_json['contents']
            name, type_json = left['name'], left['item']
            return PiType(argument=(Declaration(name=name, type=go(type_json)) if name != '_' else go(type_json)),
                          result=go(right))
        case 'App':
            head, args = type_json['contents']
            head_type = parse_head(head)
            arg_types = [go(arg) for arg in args]
            return reduce(AppType, arg_types, head_type)  # type: ignore
        case 'Lam':
            contents = type_json['contents']
            return LamType(abstraction=contents['name'], body=go(contents['item']))
        case 'Sort':
            return SortType(type_json['contents'].replace(' ', '_'))
        case 'Lit':
            return LitType(type_json['contents'].replace(' ', '_'))
        case 'Level':
            return LevelType(type_json['contents'].replace(' ', '_'))
        case _:
            raise ValueError


def parse_head(head_json: dict) -> Reference | DeBruijn:
    return Reference(name) if (name := head_json.get('Left')) is not None else DeBruijn(int(head_json.get('Right')))


def merge_scopes(file: File[str], extras: list[Declaration]) -> File[str]:
    expanded = [*extras, *[entry for entry in file.scope if entry not in extras]]
    return File(scope=expanded, holes=file.holes, name=file.name)


def enum_references(file: File[str]) -> tuple[File[int], dict[int, Name]]:
    name_to_index = defaultdict(lambda: -1, {declaration.name: idx for idx, declaration in enumerate(file.scope)})
    index_to_name = {v: k for k, v in name_to_index.items()}
    return (
        File(name=file.name,
             scope=[Declaration(name=name_to_index[declaration.name], type=declaration.type.substitute(name_to_index))
                    for declaration in file.scope],
             holes=[Hole(goal_type=hole.goal_type.substitute(name_to_index),
                         goal_term=hole.goal_term.substitute(name_to_index),
                         names_used=[name.substitute(name_to_index) for name in hole.names_used])
                    for hole in file.holes]),
        index_to_name
    )
