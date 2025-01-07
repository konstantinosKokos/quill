from json import load
from pydantic import BaseModel, ValidationError

class CachePayload(BaseModel):
    file_jsons: list[dict]


class PredictPayload(BaseModel):
    file_json: dict
    use_cache: bool


def read_json(file):
    with open(file, 'r') as f:
        return load(f)