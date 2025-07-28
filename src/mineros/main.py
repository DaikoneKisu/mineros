import pandas as pd
from sqlalchemy import create_engine, inspect

from .env import env

def load_olap_db() -> dict[str, pd.DataFrame]:
    engine = create_engine(str(env.olap_db_url))
    table_names = inspect(engine).get_table_names()

    olap_db: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        olap_db[table_name] = pd.read_sql_table(table_name, engine)
    
    return olap_db

def main():
    olap_db = load_olap_db()
    for table in olap_db:
        print(f'{table}:')
        olap_db[table].info()
        print()