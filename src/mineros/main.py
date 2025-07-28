import pandas as pd
from sqlalchemy import create_engine, inspect

from .env import env

def load_mineros_db() -> dict[str, pd.DataFrame]:
    engine = create_engine(str(env.mineros_postgres_url))
    table_names = inspect(engine).get_table_names()

    mineros_db: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        mineros_db[table_name] = pd.read_sql_table(table_name, engine)
    
    return mineros_db

def main():
    mineros_db = load_mineros_db()
    for table in mineros_db:
        print(f'{table}:')
        mineros_db[table].info()
        print()