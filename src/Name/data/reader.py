from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import reduce


@dataclass
class File:
    scope: Scope
    samples: list[Hole]


@dataclass
class Scope:
    name: str
    declarations: list[Declaration]

    def __repr__(self) -> str:
        return f'{self.name}\n' + '\n'.join('\t' + f'{d}' for d in self.declarations)

    def gather_names(self) -> set[str]: return {d.name for d in self.declarations}

    def gather_used_names(self) -> set[str]: return set.union(*[d.type.names for d in self.declarations])


@dataclass
class Hole:
    context: list[NamedType]
    goal_type: AgdaType
    pretty_type: str
    goal_term: AgdaType
    pretty_term: str
    names_used: list[Name]

    def __repr__(self) -> str:
        return '\n'.join(f'\t{c}' for c in self.context) + '\n' + ('-' * 64) + f'\n{self.goal_type}'


@dataclass
class Declaration:
    name: str
    type: AgdaType
    pretty: str

    def __repr__(self) -> str: return f'{self.name} :: {self.type}'


@dataclass
class NamedType:
    name: str
    type: AgdaType

    def __repr__(self) -> str: return f'({self.name} : {self.type})'


class AgdaType(ABC):
    @abstractmethod
    def __repr__(self) -> str: ...

    @property
    @abstractmethod
    def names(self) -> set[str]: ...


@dataclass
class PiType(AgdaType):
    nt: NamedType
    t: AgdaType

    def _rleft(self) -> str: return f'({self.nt})' if isinstance(self.nt.type, PiType) else f'{self.nt}'

    def __repr__(self) -> str: return f'{self._rleft()} -> {self.t}'

    @property
    def names(self) -> set[str]: return {self.nt.name, *self.nt.type.names, *self.t.names}


@dataclass
class LamType(AgdaType):
    abstraction: str
    body: AgdaType

    def __repr__(self) -> str: return f'λ{self.abstraction}.{self.body}'

    @property
    def names(self) -> set[str]: return {*self.body.names}


@dataclass
class AppType(AgdaType):
    head: Name | DeBruijn
    argument: AgdaType

    def __repr__(self) -> str: return f'{self.head} {self.argument}'

    @property
    def names(self) -> set[str]: return {*self.head.names, *self.argument.names}


@dataclass
class Name(AgdaType):
    string: str

    def __repr__(self) -> str: return self.string

    @property
    def names(self) -> set[str]: return {self.string}


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


def parse_data(data_json: dict) -> File:
    return File(scope=data_json['scope'], samples=[parse_sample(s) for s in data_json['samples']])


def parse_scope(scope_json: dict) -> Scope:
    name = scope_json['name']
    declarations = scope_json['item']
    return Scope(name=name, declarations=[parse_declaration(d) for d in declarations])


def parse_sample(sample_json: dict) -> Hole:
    context_json = sample_json['ctx']['thing']
    goal_type_json = sample_json['goal']
    goal_term_json = sample_json['term']
    goal_names_used = sample_json['namesUsed']

    return Hole(context=[NamedType(name=c['name'], type=parse_type(c['item'])) for c in context_json],
                goal_type=parse_type(goal_type_json['thing']),
                pretty_type=goal_type_json['pretty'],
                goal_term=parse_type(goal_term_json['thing']),
                pretty_term=goal_term_json['pretty'],
                names_used=goal_names_used)


def parse_declaration(dec_json: dict) -> Declaration:
    name = dec_json['name']
    type_json = dec_json['item']['thing']
    pretty = dec_json['item']['pretty']
    return Declaration(name=name, type=parse_type(type_json), pretty=pretty)


def parse_type(type_json: dict) -> AgdaType:
    tag = type_json['tag']
    match tag:
        case 'Pi':
            left, right = type_json['contents']
            name, type_json = left['name'], left['item']
            return PiType(nt=NamedType(name=name, type=parse_type(type_json)), t=parse_type(right))
        case 'App':
            head, args = type_json['contents']
            head_type = parse_head(head)
            arg_types = [parse_type(arg) for arg in args]
            return reduce(AppType, arg_types, head_type)
        case 'Lam':
            contents = type_json['contents']
            return LamType(abstraction=contents['name'], body=parse_type(contents['item']))
        case 'Sort':
            return SortType(type_json['contents'])
        case 'Lit':
            return LitType(type_json['contents'])
        case 'Level':
            return LevelType(type_json['contents'])
        case _:
            raise ValueError


def parse_head(head_json: dict) -> Name | DeBruijn:
    return Name(name) if (name := head_json.get('Left')) is not None else DeBruijn(int(head_json.get('Right')))
