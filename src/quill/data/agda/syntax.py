from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any
from typing_extensions import Self


Name = TypeVar('Name', str, int)
Other = TypeVar('Other')


class _AgdaExpr(ABC, Generic[Name]):
    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def substitute(self, names: dict[Name, Other]) -> Self: ...


@dataclass
class File(_AgdaExpr[Name]):
    name: str
    scope: list[ScopeEntry[Name]]
    validate: bool = True

    def __post_init__(self):
        if self.validate:
            assert self.valid_reference_structure(), 'Invalid reference structure.'
            assert self.unique_entry_names(), 'Duplicate entry names.'

    def valid_reference_structure(self) -> bool:
        names = [entry.name for entry in self.scope]
        return all(premise.name in names[:idx+1]
                   for idx, entry in enumerate(self.scope)
                   for hole in entry.holes
                   for premise in hole.premises)

    @property
    def holes(self) -> list[tuple[Name, Hole[Name]]]:
        return [(entry.name, hole) for entry in self.scope for hole in entry.holes]

    @property
    def num_holes(self) -> int:
        return len(self.holes)

    def unique_entry_names(self) -> bool:
        return len({entry.name for entry in self.scope}) == len(self.scope)

    def __repr__(self) -> str:
        return f'{self.name} ({len(self.scope)} entries)'

    def substitute(self, names: dict[Name, Other]) -> File[Other]:
        return File(name=self.name, scope=[s.substitute(names) for s in self.scope])


@dataclass
class NamedType(_AgdaExpr[Name]):
    name: Name
    type: AgdaType[Name]

    def __repr__(self) -> str:
        return f'{self.name} :: {self.type}'

    def substitute(self, names: dict[Name, Other]) -> NamedType[Other]:
        return NamedType(name=names[self.name], type=self.type.substitute(names))


@dataclass
class ScopeEntry(NamedType[Name]):
    definition: AgdaDefinition[Name]
    holes: list[Hole[Name]]
    is_import: bool

    def __repr__(self) -> str:
        return f'{super(ScopeEntry, self).__repr__()} ({len(self.holes)} holes)'

    def substitute(self, names: dict[Name, Other]) -> ScopeEntry[Other]:
        return ScopeEntry(
            name=names[self.name],
            type=self.type.substitute(names),
            definition=self.definition.substitute(names),
            holes=[hole.substitute(names) for hole in self.holes],
            is_import=self.is_import)


@dataclass(frozen=True)
class AgdaTerm(_AgdaExpr[Name], ABC):
    ...


AgdaType = AgdaTerm
Pattern = AgdaTerm


@dataclass(frozen=True)
class AgdaDefinition(_AgdaExpr[Name], ABC):
    ...


@dataclass(frozen=True)
class ADT(AgdaDefinition[Name]):
    variants: tuple[AgdaType[Name], ...]

    def __repr__(self) -> str:
        return ' | '.join(f'{v}' for v in self.variants)

    def substitute(self, names: dict[Name, Other]) -> ADT[Other]:
        return ADT(tuple(v.substitute(names) for v in self.variants))


@dataclass(frozen=True)
class Constructor(AgdaDefinition[Name]):
    reference: Name
    variant: int

    def __repr__(self) -> str:
        return f'{self.reference} ⊙ {self.variant}'

    def substitute(self, names: dict[Name, Other]) -> Constructor[Other]:
        return Constructor(reference=names[self.reference], variant=self.variant)


@dataclass(frozen=True)
class Record(AgdaDefinition[Name]):
    fields: tuple[AgdaType[Name], ...]
    telescope: tuple[AgdaType[Name], ...]

    def __repr__(self) -> str:
        return f'{self.telescope} |- {self.fields}'

    def substitute(self, names: dict[Name, Other]) -> Record[Other]:
        return Record(
            fields=tuple(f.substitute(names) for f in self.fields),
            telescope=tuple(t.substitute(names) for t in self.telescope))


@dataclass(frozen=True)
class Function(AgdaDefinition[Name]):
    clauses: tuple[FunctionClause[Name], ...]

    def __repr__(self) -> str:
        return '\n'.join(f'{clause}' for clause in self.clauses)

    def substitute(self, names: dict[Name, Other]) -> Function[Other]:
        return Function(tuple(c.substitute(names) for c in self.clauses))


@dataclass(frozen=True)
class FunctionClause(_AgdaExpr[Name]):
    telescope: tuple[NamedType[Name], ...]
    patterns: tuple[Pattern[Name], ...]
    body: Optional[AgdaTerm[Name]]

    def __repr__(self) -> str:
        return f'{self.telescope} |- {self.patterns} = {self.body}'

    def substitute(self, names: dict[Name, Other]) -> FunctionClause[Other]:
        return FunctionClause(
            telescope=tuple(t.substitute(names) for t in self.telescope),
            patterns=tuple(p.substitute(names) for p in self.patterns),
            body=self.body.substitute(names) if self.body is not None else None)


@dataclass(frozen=True)
class Postulate(AgdaDefinition[Name]):
    def __repr__(self) -> str: return 'Postulate'
    def substitute(self, names: dict[Name, Other]) -> Postulate[Other]: return Postulate()


@dataclass(frozen=True)
class Primitive(AgdaDefinition[Name]):
    def __repr__(self) -> str: return 'Primitive'
    def substitute(self, names: dict[Name, Other]) -> Primitive[Other]: return Primitive()


@dataclass(frozen=True)
class Hole(_AgdaExpr[Name]):
    context: tuple[AgdaType, ...]       # n.b. missing context names
    goal: AgdaType[Name]
    term: AgdaTerm[Name]
    premises: tuple[Reference[Name], ...]     # n.b. scope only, above or self

    def __repr__(self) -> str:
        return f'{self.context} |- {self.goal}'

    def substitute(self, names: dict[Name, Other]) -> Hole[Other]:
        return Hole(
            context=tuple(c.substitute(names) for c in self.context),
            goal=self.goal.substitute(names),
            term=self.term.substitute(names),
            premises=tuple(premise.substitute(names) for premise in self.premises))


@dataclass(frozen=True)
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
        return PiTerm(
            domain=self.domain.substitute(names),
            codomain=self.codomain.substitute(names),
            name=None if self.name is None else names[self.name])


@dataclass(frozen=True)
class LamTerm(AgdaTerm[Name]):
    abstraction: Any
    body: AgdaTerm[Name]

    def __repr__(self) -> str: return f'λ{self.abstraction}.{self.body}'

    def substitute(self, names: dict[Name, Other]) -> LamTerm[Other]:
        return LamTerm(self.abstraction, self.body.substitute(names))


@dataclass(frozen=True)
class AppTerm(AgdaTerm[Name]):
    head: Reference[Name] | DeBruijn
    argument: AgdaTerm[Name]

    def __repr__(self) -> str:
        arg_repr = f'({self.argument})' if isinstance(self.argument, AppTerm) else f'{self.argument}'
        return f'{self.head} {arg_repr}'

    def substitute(self, names: dict[Name, Other]) -> AppTerm[Other]:
        return AppTerm(self.head.substitute(names), self.argument.substitute(names))


@dataclass(frozen=True)
class Reference(AgdaTerm[Name]):
    name: Name

    def __repr__(self) -> str: return f'{self.name}'

    def substitute(self, names: dict[Name, Other]) -> Reference[Other]:
        return Reference(names[self.name])


@dataclass(frozen=True)
class DeBruijn(AgdaTerm[Name]):
    index: int

    def __repr__(self) -> str: return f'@{self.index}'

    def substitute(self, names: dict[Name, Other]) -> DeBruijn[Other]:
        return DeBruijn(self.index)


@dataclass(frozen=True)
class LitTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LitTerm[Other]:
        return LitTerm(self.content)


@dataclass(frozen=True)
class SortTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> SortTerm[Other]:
        return SortTerm(self.content)


@dataclass(frozen=True)
class LevelTerm(AgdaTerm[Name]):
    content: Any

    def __repr__(self) -> str: return f'{self.content}'

    def substitute(self, names: dict[Name, Other]) -> LevelTerm[Other]:
        return LevelTerm(self.content)


@dataclass(frozen=True)
class UnsolvedMeta(AgdaTerm[Name]):
    def __repr__(self) -> str: return '{}'
    def substitute(self, names: dict[Name, Other]) -> UnsolvedMeta[Other]: return UnsolvedMeta()
