from __future__ import annotations

from json import load
from os import listdir, path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeVar, Any
from typing_extensions import Self

Name = TypeVar('Name')
Other = TypeVar('Other')


@dataclass(unsafe_hash=True)
class File(Generic[Name]):
    name: str
    scope: list[Declaration[Name]]
    samples: list[Hole[Name]]

    def __repr__(self) -> str:
        return '\n'.join(f'{d}' for d in self.scope) + f'\n{"="*64}\n\n' + '\n'.join(f'{s}' for s in self.samples)


@dataclass(unsafe_hash=True)
class Hole(Generic[Name]):
    context: list[Declaration[Name]]
    goal_type: AgdaType[Name]
    goal_term: AgdaType[Name]
    names_used: list[Reference[Name]]  # n.b. scope reference only

    def __repr__(self) -> str:
        return '\n'.join(f'\t{c}' for c in self.context) + f'\n\n\t{self.goal_term} : {self.goal_type}\n' + ('-' * 64)


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


Telescope = tuple[Declaration, ...]


@dataclass(unsafe_hash=True)
class PiType(AgdaType[Name]):
    argument: AgdaType[Name]
    result: AgdaType[Name]

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, (Declaration, PiType)) else f'{self.argument}'
        return f'{arg_repr} -> {self.result}'

    def substitute(self, names: dict[Name, Other]) -> PiType[Other]:
        return PiType(self.argument.substitute(names), self.result.substitute(names))


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


def parse_dir(directory: str) -> list[File]:
    return [parse_file(path.join(directory, file)) for file in listdir(directory)]


def parse_file(filepath: str) -> File:
    with open(filepath, 'r') as f:
        return parse_data(load(f))


def parse_data(data_json: dict) -> File:
    return File(name=data_json['scope']['name'],
                scope=[parse_declaration(d) for d in data_json['scope']['item']],
                samples=[parse_sample(s) for s in data_json['samples']])


def parse_sample(sample_json: dict) -> Hole:
    context_json = sample_json['ctx']['thing']
    goal_type_json = sample_json['goal']
    goal_term_json = sample_json['term']
    goal_names_used = sample_json['namesUsed']

    return Hole(context=[Declaration(name=c['name'], type=parse_type(c['item'])) for c in context_json],
                goal_type=parse_type(goal_type_json['thing']),
                goal_term=parse_type(goal_term_json['thing']),
                names_used=[Reference(name) for name in goal_names_used])


def parse_declaration(dec_json: dict) -> Declaration:
    return Declaration(name=dec_json['name'], type=parse_type(dec_json['item']['thing']))


def parse_type(type_json: dict) -> AgdaType:
    match type_json['tag']:
        case 'Pi':
            left, right = type_json['contents']
            name, type_json = left['name'], left['item']
            return PiType(argument=(Declaration(name=name, type=parse_type(type_json))
                                    if name != '_' else parse_type(type_json)),
                          result=parse_type(right))
        case 'App':
            head, args = type_json['contents']
            head_type = parse_head(head)
            arg_types = [parse_type(arg) for arg in args]
            return reduce(AppType, arg_types, head_type)  # type: ignore
        case 'Lam':
            contents = type_json['contents']
            return LamType(abstraction=contents['name'], body=parse_type(contents['item']))
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
