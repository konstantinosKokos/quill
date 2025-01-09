def main(
    config_path: str,
    weight_path: str,
    device: str,
    host: str,
    port: int,
):
    import uvicorn
    from fastapi import FastAPI


    from .utils import PredictPayload, CachePayload, read_json
    from ..nn.inference import Inferer
    from ..data.agda.reader import parse_data

    print(f'Initializing model from {config_path}...')
    inferer = Inferer(model_config=read_json(config_path), cast_to=device).eval()
    print(f'Loading weights from {weight_path}...')
    inferer.load(path=weight_path, strict=True, map_location=device)
    print('Done.')

    app = FastAPI()
    @app.post('/cache', response_model=int)
    async def precompute(payload: CachePayload):
        inferer.precompute(files=[parse_data(f, validate=True) for f in payload.file_jsons])
        return len(inferer.cache)

    @app.post('/predict', response_model=list[list[str]])
    async def predict(payload: PredictPayload):
        return inferer.select_premises(file=parse_data(payload.file_json, validate=True), use_cache=payload.use_cache)

    uvicorn.run(app, host=host, port=port, reload=False)
