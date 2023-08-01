"""
 Processes the Agda2Train extraction JSON into human-readable format.
"""
import pdb
from functools import reduce


def beautify_file(file_json: dict) -> dict:
    return {'scope': beautify_scope(file_json['scope']),
            'goals': [beautify_hole(hole) for hole in file_json['samples']]}


def beautify_scope(scope_json: dict) -> dict:
    return {'entries': [beautify_scope_entry(entry) for entry in scope_json['item']]}


def beautify_scope_entry(entry_json: dict) -> dict:
    return {'name': entry_json['name'],
            'type': beautify_top_term(entry_json['item'][0]),
            'definition': beautify_top_term(entry_json['item'][1])}


def beautify_hole(hole_json: dict) -> dict:
    return {'definition': beautify_top_term(hole_json['term']),
            'type': {'pretty': hole_json['ctx']['pretty'],
                     'term': reduce(lambda result, argument:
                                    {'tag': 'Pi',
                                     'name': argument['name'],
                                     'domain': beautify_term(argument['item']),
                                     'codomain': result},
                                    reversed(hole_json['ctx']['thing']),
                                    beautify_term(hole_json['goal']['thing']['original']))},
            'premises': hole_json['namesUsed']}


def beautify_top_term(term_json: dict) -> dict:
    inner = term_json['thing']
    if 'original' in inner:
        inner = inner['original']
    return {'pretty': term_json['pretty'],
            'term': beautify_term(inner)}


def beautify_term(term_json: dict) -> dict:
    match term_json['tag']:
        case 'Pi':
            domain, codomain = term_json['contents']
            return {'tag': 'Pi',
                    'name': domain['name'],
                    'domain': beautify_term(domain['item']),
                    'codomain': beautify_term(codomain)}
        case 'App':
            head, args = term_json['contents']
            if head == {'Left': '⊕'}:
                return {'tag': 'ADT',
                        'variants': [beautify_term(arg) for arg in args]}
            if head == {'Left': '⊙'}:
                return {'tag': 'Constructor',
                        'name': args[0]['contents'][0]['Left'],
                        'variant': args[1]['contents']}
            return {'tag': 'App',
                    'head': beautify_head(head),
                    'arguments': [beautify_term(arg) for arg in args]}
        case 'Lam':
            return {'tag': 'Lam',
                    'body': beautify_term(term_json['contents']['item']),
                    'abstraction': term_json['contents']['name']}
        case 'Sort':
            return {'tag': 'Sort',
                    'sort': term_json['contents']}
        case 'Lit':
            return {'tag': 'Literal',
                    'literal': term_json['contents']}
        case 'Level':
            return {'tag': 'Level',
                    'level': term_json['contents']}
    raise ValueError(term_json['tag'])


def beautify_head(head_json: dict) -> dict:
    if 'Left' in head_json:
        return {'tag': 'ScopeReference', 'name': head_json['Left']}
    if 'Right' in head_json:
        return {'tag': 'deBruijn', 'index': head_json['Right']}
    raise ValueError
