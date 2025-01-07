def main(
    config_path: str,
    weight_path: str,
    device: str,
    host: str,
    port: int,
):
    from flask import Flask, request, jsonify

    from .schemas import PredictPayload, CachePayload, ValidationError, read_json
    from ..nn.inference import Inferer
    from ..data.agda.reader import parse_data

    print(f'Initializing model from {config_path}...')
    inferer = Inferer(model_config=read_json(config_path), cast_to=device).eval()
    print(f'Loading weights from {weight_path}...')
    inferer.load(path=weight_path, strict=True, map_location=device)
    print('Done.')

    app = Flask(__name__)
    @app.route('/cache', methods=['POST'])
    def precompute():
        try:
            payload = CachePayload(**request.json)
        except ValidationError as e:
            return jsonify({'error': e.errors()}), 400
        if len(payload.file_jsons) > 0:
            inferer.precompute(files=[parse_data(f, validate=True) for f in payload.file_jsons])
            return jsonify({'message': f'Cache updated with {len(inferer.cache)} lemmas'}), 200
        inferer.cache = []
        return jsonify({'message': f'Cache emptied'}), 200


    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            payload = PredictPayload(**request.json)
        except ValidationError as e:
            return jsonify({'error': e.errors()}), 400
        output = inferer.select_premises(file=parse_data(payload.file_json, validate=True), use_cache=payload.use_cache)
        return jsonify(output)

    app.run(debug=False, host=host, port=port)
    print(f'Serving on {host}:{port}')
