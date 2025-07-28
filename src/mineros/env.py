from pprint import pprint
import sys

from pydantic import AnyUrl, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv(override=True)

class _Env(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
    )

    mineros_postgres_url: AnyUrl

env: _Env
try:
    env = _Env() # type: ignore
except ValidationError as e:
    print('An error occurred while parsing the .env file:')
    pprint(e.errors())
    sys.exit(1)
