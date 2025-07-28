from pathlib import Path
from pprint import pprint
import sys

from pydantic import AnyUrl, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

class _Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(Path(__file__).parent.parent.parent / '.env').absolute(),
        extra='ignore',
    )

    olap_db_url: AnyUrl

env: _Env
try:
    env = _Env() # type: ignore
except ValidationError as e:
    print('An error occurred while parsing the .env file:')
    pprint(e.errors())
    sys.exit(1)
