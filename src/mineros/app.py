import streamlit as st
import pandas as pd
import plotly.express as px # pyright: ignore[reportMissingTypeStubs]
from mineros import load_data 

# Configuración de la página
st.set_page_config(
    page_title="Dashboard EDA de Sesiones de Juego",
    page_icon="🎮",
    layout="wide"
)


# Cargar los datos
df = load_data()


# --- Título del Dashboard ---
st.title("🎮 Dashboard de Análisis Exploratorio de Sesiones de Juego")
st.markdown("Análisis interactivo de la actividad de juego basado en el modelo OLAP.")


# --- Sidebar de Filtros ---
st.sidebar.header("Filtros 🔎")
selected_providers = st.sidebar.multiselect(
    "Selecciona Proveedores",
    options=df['provider_name'].unique(),
    default=df['provider_name'].unique()
)

selected_game_types = st.sidebar.multiselect(
    "Selecciona Tipos de Juego",
    options=df['game_type'].unique(),
    default=df['game_type'].unique()
)

age_range = st.sidebar.slider(
    "Selecciona Rango de Edad del Jugador",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max()))
)

# Filtrar el DataFrame basado en la selección del sidebar
df_filtered = df[
    (df['provider_name'].isin(selected_providers)) & # pyright: ignore[reportUnknownMemberType]
    (df['game_type'].isin(selected_game_types)) & # pyright: ignore[reportUnknownMemberType]
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1])
]

# --- Dashboard Principal ---

# Métricas Clave (KPIs)
st.header("Métricas Principales")
total_amount = df_filtered['amount'].sum()
total_sessions = df_filtered['session_id'].count()
average_amount = df_filtered['amount'].mean()
unique_players = df_filtered['player_name'].nunique()

col1, col2 = st.columns(2)
col1.metric("Monto Total", f"VES {total_amount:,.2f}")
col1.metric("Monto Promedio/Sesión", f"VES {average_amount:,.2f}")
col2.metric("Sesiones Totales", f"{total_sessions:,}")
col2.metric("Jugadores Únicos", f"{unique_players:,}")

st.markdown("---")

# Visualizaciones
st.header("Análisis Visual")
col1, col2 = st.columns(2)

with col1:
    # Gráfico de barras: Monto total por proveedor
    st.subheader("Monto Total por Proveedor")
    amount_by_provider = df_filtered.groupby('provider_name')['amount'].sum().sort_values(ascending=False).reset_index() # pyright: ignore[reportUnknownMemberType]
    fig1 = px.bar(amount_by_provider, # pyright: ignore[reportUnknownMemberType]
                  x='provider_name',
                  y='amount',
                  title="Ingresos por Proveedor",
                  labels={'provider_name': 'Proveedor', 'amount': 'Monto Total (VES)'},
                  template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True) # pyright: ignore[reportUnknownMemberType]

    # Gráfico de pastel: Distribución por tipo de juego
    st.subheader("Popularidad de Tipos de Juego")
    sessions_by_type = df_filtered['game_type'].value_counts().reset_index()
    fig2 = px.pie(sessions_by_type, # pyright: ignore[reportUnknownMemberType]
                  names='game_type',
                  values='count',
                  title="Distribución de Sesiones por Tipo de Juego",
                  labels={'game_type': 'Tipo de Juego', 'count': 'Número de Sesiones'},
                  hole=0.3)
    st.plotly_chart(fig2, use_container_width=True) # pyright: ignore[reportUnknownMemberType]

with col2:
    # Gráfico de línea: Monto total a lo largo del tiempo
    st.subheader("Monto Total por Día")
    amount_over_time = df_filtered.set_index('datetime').resample('D')['amount'].sum().reset_index()
    fig3 = px.line(amount_over_time, # pyright: ignore[reportUnknownMemberType]
                   x='datetime',
                   y='amount',
                   title="Tendencia de Monto Jugado por Día",
                   labels={'datetime': 'Fecha', 'amount': 'Monto Total (VES)'},
                   template='plotly_white')
    fig3.update_traces(mode='lines+markers') # pyright: ignore[reportUnknownMemberType]
    st.plotly_chart(fig3, use_container_width=True) # pyright: ignore[reportUnknownMemberType]

    # Histograma: Distribución de edad de los jugadores
    st.subheader("Distribución de Edad de Jugadores")
    fig4 = px.histogram(df_filtered, # pyright: ignore[reportUnknownMemberType]
                        x='age',
                        nbins=20,
                        title="Frecuencia por Rango de Edad",
                        labels={'age': 'Edad', 'count': 'Cantidad de Jugadores'},
                        template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True) # pyright: ignore[reportUnknownMemberType]


# Vista detallada de los datos
st.header("Explorar Datos Detallados")
with st.expander("Haz clic para ver la tabla de datos filtrados"):
    st.dataframe(df_filtered) # pyright: ignore[reportUnknownMemberType]