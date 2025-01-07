from argparse import ArgumentParser
from flask import Flask, request, jsonify
import json
from quill.nn.inference import Inferer
from quill.data.agda.reader import parse_data



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-config', type=str, help="Path to the model config file")
    parser.add_argument('-weights', type=str, help="Path to the model weights file")
    parser.add_argument('--device', type=str, choices=('cuda', 'cpu'), help="Device to run inference on", default='cpu')
    parser.add_argument('--host', type=str, help='Server host address', default='127.0.0.1')
    parser.add_argument('--port', type=str, help='Server host post', default='5000')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Initializing model...')
    inferer = Inferer(model_config=json.load(open(args.config, 'r')), cast_to=args.device)
    print('Loading weights...')
    inferer.load(path=args.weights, strict=True, map_location=args.device)
    inferer.eval()
    print('Done.')

    app = Flask(__name__)
    @app.route('/cache', methods=['POST'])
    def cache():
        payload = request.json
        inferer.precompute(files=[parse_data(f, validate=True) for f in payload['files']])


    @app.route('/predict', methods=['POST'])
    def predict():
        payload = request.json
        output = inferer.select_premises(file=parse_data(payload['file'], validate=True), use_cache=payload['use_cache'])
        return jsonify(output)

    app.run(debug=False, host=args.host, port=args.port)
