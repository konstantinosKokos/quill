from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Optional
from typing_extensions import Self


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
    lemmas: set[Reference[Name]]  # n.b. scope reference only

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
