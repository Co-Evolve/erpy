import dataclasses
import json
from typing import Dict


class Config2JSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)
        except TypeError:
            return str(o)


def config2json(config: dataclasses.dataclass) -> str:
    return json.dumps(obj=config, cls=Config2JSONEncoder)


def config2dict(config: dataclasses.dataclass) -> Dict:
    return json.loads(config2json(config=config))
