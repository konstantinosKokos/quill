def main(
    file: str,
    host: str,
    port: int,
    use_cache: bool,
    max_suggestions: int | None,
):
    import requests
    from .utils import PredictPayload, read_json

    url = f'http://{host}:{port}/predict'
    payload = PredictPayload(file_json=read_json(file), use_cache=use_cache)
    response = requests.post(url=url, json=payload.model_dump())

    if response.status_code == 200:
        suggestions = [xs[:max_suggestions] for xs in response.json()]
        print(suggestions)
    else:
        print(f'Received status code {response.status_code}: {response.text}')
