from __future__ import annotations

from json import load
from os import listdir, path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, TypeVar


Name = TypeVar('Name')
Other = TypeVar('Other')


@dataclass
class File:
    name: str
    scope: list[Declaration]
    samples: list[Hole]

    def __repr__(self) -> str:
        return '\n'.join(f'{d}' for d in self.scope) + f'\n{"="*64}\n\n' + '\n'.join(f'{s}' for s in self.samples)


@dataclass
class Hole:
    context: list[Declaration]
    goal_type: AgdaType
    goal_term: AgdaType
    names_used: list[Reference]

    def __repr__(self) -> str:
        return '\n'.join(f'\t{c}' for c in self.context) + f'\n\n\t{self.goal_term} : {self.goal_type}\n' + ('-' * 64)


@dataclass
class Declaration(Generic[Name]):
    name: Name
    type: AgdaType

    def __repr__(self) -> str: return f'{self.name} :: {self.type}'


class AgdaType(ABC):
    @abstractmethod
    def __repr__(self) -> str: ...

    @property
    @abstractmethod
    def names(self) -> set[str]: ...


@dataclass
class PiType(AgdaType):
    argument: AgdaType | Declaration
    result: AgdaType

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, (Declaration, PiType)) else f'{self.argument}'
        return f'{arg_repr} -> {self.result}'

    @property
    def names(self) -> set[str]: return {self.argument.name, *self.argument.type.names, *self.result.names}


@dataclass
class LamType(AgdaType):
    abstraction: str
    body: AgdaType

    def __repr__(self) -> str: return f'λ{self.abstraction}.{self.body}'

    @property
    def names(self) -> set[str]: return {*self.body.names}


@dataclass
class AppType(AgdaType):
    head: Reference | DeBruijn
    argument: AgdaType

    def __repr__(self) -> str: return f'{self.head} {self.argument}'

    @property
    def names(self) -> set[str]: return {*self.head.names, *self.argument.names}


@dataclass
class Reference(AgdaType, Generic[Name]):
    name: Name

    def __repr__(self) -> str: return self.name

    @property
    def names(self) -> set[str]: return {self.name}


@dataclass
class DeBruijn(AgdaType):
    index: int

    def __repr__(self) -> str: return f'@{self.index}'

    @property
    def names(self) -> set[str]: return set()


@dataclass
class LitType(AgdaType):
    string: str

    def __repr__(self) -> str: return self.string

    @property
    def names(self) -> set[str]: return set()


@dataclass
class SortType(AgdaType):
    string: str

    def __repr__(self) -> str: return self.string

    @property
    def names(self) -> set[str]: return set()


@dataclass
class LevelType(AgdaType):
    string: str

    def __repr__(self) -> str: return self.string

    @property
    def names(self) -> set[str]: return set()


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
