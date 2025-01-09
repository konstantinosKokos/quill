from argparse import ArgumentParser

def main():
    parser = ArgumentParser(prog='quill')
    subparsers = parser.add_subparsers(dest='command', required=True)

    serve_parser = subparsers.add_parser('serve', help='Start the model server')
    serve_parser.add_argument('-config', type=str, required=True, help="Path to the model config file")
    serve_parser.add_argument('-weights', type=str, required=True, help="Path to the model weights file")
    serve_parser.add_argument('--device', type=str, choices=('cuda', 'cpu'), default='cpu', help="Device to run inference on")
    serve_parser.add_argument('--host', type=str, default='127.0.0.1', help="Server host address")
    serve_parser.add_argument('--port', type=int, default=5000, help="Server port")

    query_parser = subparsers.add_parser('query', help='Query the model with an Agda export file')
    query_parser.add_argument('-file', type=str, required=True, help='Path to an Agda json export')
    query_parser.add_argument('--max_suggestions', type=int, default=None, help='Max number of premises returned per hole')
    query_parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    query_parser.add_argument('--port', type=int, default=5000, help='Server port')
    query_parser.add_argument('--use_cache', action='store_true', help='Suggest lemmas outside the current scope')

    cache_parser = subparsers.add_parser('cache', help='Cache lemmas from a collection of files')
    cache_parser.add_argument('-files', type=str, nargs='*', required=True, help='Paths to Agda export files')
    cache_parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    cache_parser.add_argument('--port', type=int, default=5000, help='Server port')

    args = parser.parse_args()

    match args.command:
        case 'serve':
            from .serve import main
            main(
                config_path=args.config,
                weight_path=args.weights,
                device=args.device,
                host=args.host,
                port=args.port
            )
        case 'query':
            from .query import main
            main(
                file=args.file,
                max_suggestions=args.max_suggestions,
                host=args.host,
                port=args.port,
                use_cache=args.use_cache,
            )
        case 'cache':
            from .cache import main
            main(
                files=args.files,
                host=args.host,
                port=args.port,
            )
        case _:
            raise ValueError(f'Unrecognized command {args.command}')
