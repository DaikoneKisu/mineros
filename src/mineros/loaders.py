import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from .env import env

def load_mineros_db() -> dict[str, pd.DataFrame]:
    engine = create_engine(str(env.mineros_postgres_url))
    table_names = inspect(engine).get_table_names(env.mineros_postgres_schema)

    mineros_db: dict[str, pd.DataFrame] = {}
    for table_name in table_names:
        print(f"Loading table: {table_name}, starting at timestamp: {pd.Timestamp.now()}")
        mineros_db[table_name] = pd.read_sql_table(table_name, engine, schema=env.mineros_postgres_schema)
        print(f"Loaded table: {table_name}, ending at timestamp: {pd.Timestamp.now()}")

    return mineros_db

@st.cache_data
def load_data():
    dfs = load_mineros_db()

    df_dim_provider = dfs["dim_provider"]
    df_dim_game = dfs["dim_game"]
    df_dim_player = dfs["dim_player"]
    df_fact_game_session = dfs["fact_game_session"]

    df = pd.merge(df_fact_game_session, df_dim_game, left_on='game_id', right_on='id', suffixes=('', '_game'))
    df = pd.merge(df, df_dim_player, left_on='player_id', right_on='id', suffixes=('', '_player'))
    df = pd.merge(df, df_dim_provider, left_on='provider_id', right_on='id', suffixes=('', '_provider'))

    df.rename(columns={'name': 'game_name', 'name_player': 'player_name', 'name_provider': 'provider_name', 'type': 'game_type', 'id': 'session_id'}, inplace=True)
    df.drop(['id_game', 'id_player', 'id_provider', 'game_id', 'player_id', 'provider_id'], axis=1, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df

@st.cache_data
def load_models_data():
    dfs = load_mineros_db()

    df_dim_provider = dfs["dim_provider"]
    df_dim_game = dfs["dim_game"]
    df_dim_player = dfs["dim_player"]
    df_fact_game_session = dfs["fact_game_session"]

    df = pd.merge(df_fact_game_session, df_dim_game, left_on='game_id', right_on='id', suffixes=('', '_game'))
    df = pd.merge(df, df_dim_player, left_on='player_id', right_on='id', suffixes=('', '_player'))
    df = pd.merge(df, df_dim_provider, left_on='provider_id', right_on='id', suffixes=('', '_provider'))
    df.rename(columns={'name': 'game_name', 'name_player': 'player_name', 'name_provider': 'provider_name', 'type': 'game_type', 'id': 'session_id'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # --- Feature Engineering para los modelos ---
    player_summary = df.groupby('player_id').agg(
        total_spent=('amount', 'sum'),
        total_sessions=('session_id', 'count'),
        first_session=('datetime', 'min'),
        last_session=('datetime', 'max'),
        favorite_game_type=('game_type', lambda x: x.mode()[0]),
        favorite_provider=('provider_name', lambda x: x.mode()[0])
    ).reset_index()

    # Calcular retenci贸n y gasto futuro (simulado)
    player_summary['retention_days'] = (player_summary['last_session'] - player_summary['first_session']).dt.days
    player_summary['future_spend'] = player_summary['total_spent'] * np.random.uniform(0.1, 0.5) # Simulaci贸n

    # Unir con datos demogr谩ficos
    final_df = pd.merge(player_summary, df_dim_player, left_on='player_id', right_on='id')
    
    return df, final_df
