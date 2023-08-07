"""
 Processes the Agda2Train extraction JSON into human-readable format.
"""
from functools import reduce


def beautify_file(file_json: dict) -> dict:
    return {'name': file_json['scope']['name'],
            'scope': beautify_scope(file_json['scope']),
            'holes': [beautify_hole(hole) for hole in file_json['samples']]}


def beautify_scope(scope_json: dict) -> dict:
    return {'entries': [beautify_scope_entry(entry) for entry in scope_json['item']]}


def beautify_scope_entry(entry_json: dict) -> dict:
    return {'name': entry_json['name'],
            'type': beautify_top_term(entry_json['item'][0]),
            'definition': beautify_top_term(entry_json['item'][1])}


def beautify_hole(hole_json: dict) -> dict:
    return {'definition': beautify_top_term(hole_json['term']),
            'type': {'pretty': f'{hole_json["ctx"]["pretty"]} → {hole_json["goal"]["pretty"]}',
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
    return {'pretty': term_json['pretty'].replace('"', ''),
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
                        'reference': args[0]['contents'][0]['Left'],
                        'variant': args[1]['contents']}
            return {'tag': 'Application',
                    'head': beautify_head(head),
                    'arguments': [beautify_term(arg) for arg in args]}
        case 'Lam':
            return {'tag': 'Lambda',
                    'body': beautify_term(term_json['contents']['item']),
                    'abstraction': term_json['contents']['name'].replace('"', '')}
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


def main(input_directory: str, output_directory: str):
    from json import load, dumps
    from os import listdir, path

    errors = {}

    for file in listdir(input_directory):
        if path.isfile(path.join(input_directory, file)):
            with open(path.join(input_directory, file), 'r') as f:
                try:
                    out = beautify_file(load(f))
                except Exception as e:
                    errors[file] = repr(e)
                    continue
            with open(path.join(output_directory, file), 'w') as f:
                f.write(dumps(out, indent=4, ensure_ascii=False))

    with open(path.join(output_directory, 'errors'), 'w') as f:
        f.write(dumps(errors, indent=4, ensure_ascii=False))
