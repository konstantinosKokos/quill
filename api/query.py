from argparse import ArgumentParser
from json import load
import requests


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-file', type=str, help='Path to an Agda json export')
    parser.add_argument('--host', type=str, help='Server host address', default='127.0.0.1')
    parser.add_argument('--port', type=str, help='Server host port', default='5000')
    parser.add_argument('--use_cache', action='store_true', help='Suggest lemmas outside the current scope')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    url = f'http://{args.host}:{args.port}/predict'

    with open(args.file, 'r') as f:
        payload = {'file': load(f), 'use_cache': args.use_cache}

    response = requests.post(url=url, json=payload)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f'Received status code {response.status_code}')
        print(response.content)