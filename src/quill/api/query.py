def main(
    file: str,
    host: str,
    port: int,
    use_cache: bool
):
    import requests
    from json import load
    from .schemas import PredictPayload, read_json

    url = f'http://{host}:{port}/predict'
    payload = PredictPayload(file_json=read_json(file), use_cache=use_cache)
    response = requests.post(url=url, json=payload.model_dump())
    if response.status_code == 200:
        print(response.json())
    else:
        print(f'Received status code {response.status_code}')
        print(response.content)