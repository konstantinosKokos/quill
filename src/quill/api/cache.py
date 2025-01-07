def main(
    files: list[str],
    host: str,
    port: int
):
    import requests
    from .schemas import CachePayload, read_json

    url = f'http://{host}:{port}/cache'
    payload = CachePayload(file_jsons=list(map(read_json, files)))
    response = requests.post(url=url, json=payload.model_dump())
    if response.status_code == 200:
        print(response.json()['message'])
    else:
        print(f'Received status code {response.status_code}')
        print(response.json()['error'])