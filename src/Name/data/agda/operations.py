from .syntax import File, ScopeEntry, Name, AgdaTerm, PiTerm, LamTerm, AppTerm, Reference, Hole, UnsolvedMeta

from collections import defaultdict
from functools import reduce


def enum_references(file: File[Name]) -> tuple[File[int], dict[int, Name]]:
    name_to_index = defaultdict(lambda: -1, {declaration.name: idx for idx, declaration in enumerate(file.scope)})
    index_to_name = {v: k for k, v in name_to_index.items()}
    file = File(
        name=file.name,
        scope=[
            ScopeEntry(
                name=name_to_index[entry.name],
                type=entry.type.substitute(name_to_index),
                definition=entry.definition.substitute(name_to_index),
                holes=[hole.substitute(name_to_index) for hole in entry.holes],
                is_import=entry.is_import)
            for entry in file.scope])
    return file, index_to_name


def get_references(term: AgdaTerm) -> set[Name]:
    match term:
        case PiTerm(domain, codomain, _):
            return get_references(domain) | get_references(codomain)
        case LamTerm(_, body):
            return get_references(body)
        case AppTerm(head, argument):
            return get_references(head) | get_references(argument)
        case Reference(name):
            return {name}
        case _:
            return set()


def top_sort(entries: dict[Name, AgdaTerm], pre_resolved: dict[Name, int]) -> dict[Name, int]:
    if common := entries.keys() & pre_resolved.keys():
        raise AssertionError(f'Common names: {common}')
    referrable = entries.keys() | pre_resolved.keys()
    references = {name: get_references(term).intersection(referrable)
                  for name, term in entries.items()}
    levels = {**{name: None for name in entries.keys()},
              **{name: level for name, level in pre_resolved.items()}}

    resolved = set(pre_resolved.keys())
    current = max(pre_resolved.values(), default=0)
    while unresolved := {k for k, v in levels.items() if v is None}:
        resolvable = {k for k in unresolved if references[k].issubset(resolved)}
        resolved |= resolvable
        for key in resolvable:
            levels[key] = current
        current += 1
    return {entry: levels[entry] for entry in entries}


def top_sort_entries(file: File[Name]) -> dict[Name, int]:
    return top_sort(
        entries={entry.name: entry.type for entry in file.scope},
        pre_resolved=dict())


def merge_contexts(file: File[Name], merge_holes: bool, unique_only: bool, validate: bool = True) -> File[Name]:
    def ctx_to_pi(hole: Hole[Name]) -> Hole[Name]:
        return Hole(
            context=(),
            goal=reduce(lambda cod, dom: PiTerm(dom[1], cod, f'ctx_{dom[0]}'),
                        reversed(tuple(enumerate(hole.context))),
                        hole.goal),  # type: ignore
            term=UnsolvedMeta(),
            premises=hole.premises
        )

    def f(holes: list[Hole[Name]]) -> list[Hole[Name]]:
        if merge_holes:
            holes = _merge_holes(holes)
        if unique_only:
            holes = _keep_unique_holes(holes)
        return holes

    return File(
        name=file.name,
        scope=[
            ScopeEntry(
                name=entry.name,
                type=entry.type,
                definition=entry.definition,
                holes=f([ctx_to_pi(h) for h in entry.holes]),
                is_import=entry.is_import)
            for entry in file.scope],
        validate=validate)


def _merge_holes(holes: list[Hole[Name]]) -> list[Hole[Name]]:
    unique_types = {(hole.context, hole.goal) for hole in holes}
    return [Hole(
        goal=goal,
        premises=tuple(p for h in holes if h.goal == goal and h.context == ctx for p in h.premises),
        context=ctx,
        term=UnsolvedMeta()) for ctx, goal in unique_types]


def _keep_unique_holes(holes: list[Hole[Name]]) -> list[Hole[Name]]:
    return list(set(holes))
