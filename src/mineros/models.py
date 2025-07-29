import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Modelos y preprocesamiento ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from mineros import load_models_data, k_prototypes_fit, find_optimal_k

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Pipeline de Modelos de Machine Learning",
    page_icon="🎮",
    layout="wide"
)

# --- Cargar Datos ---
df_transactions, df_players = load_models_data()
df_transactions = df_transactions.rename(str, axis='columns')
df_players = df_players.rename(str, axis='columns')

# --- Título Principal ---
st.title("🎮 Pipeline de Modelos de Machine Learning para Gaming")
st.markdown("Una aplicación para explorar segmentación de clientes, reglas de asociación y modelos predictivos.")

# --- Pestañas para cada modelo ---
tab1, tab2, tab3 = st.tabs(["🎭 Agrupamiento (K-Prototypes)", "📊 Reglas de Asociación (Apriori)", "🔮 Predicción (Random Forest)"])

# =================================================================================================
# PESTAÑA 1: K-PROTOTYPES
# =================================================================================================
with tab1:
    st.header("Segmentación de Jugadores con K-Prototypes")

    with st.expander("Justificación del Modelo", expanded=True):
        st.markdown("""
        El objetivo es segmentar usuarios en perfiles de comportamiento, lo cual es un problema de agrupamiento (clustering). El algoritmo **K-Prototypes** es la elección ideal para esta tarea porque el conjunto de datos de los jugadores contiene una mezcla de tipos de datos. 
        - **Variables Numéricas:** `edad`, `gasto total`, `días de retención`.
        - **Variables Categóricas:** `tipo de juego preferido`, `proveedor preferido`.

        K-Prototypes combina K-Means (para numéricas) y K-Modes (para categóricas), permitiendo agrupar a los usuarios de manera coherente y holística para crear perfiles robustos y significativos.
        """)

    st.subheader("1. Entrenamiento y Validación")
    st.markdown("Para encontrar el número óptimo de clústeres (k), usamos el **Método del Codo (Elbow Method)**, que mide el costo (disimilitud) para diferentes valores de k.")

    # Preparación de datos para K-Prototypes
    data_for_clustering = df_players[['age', 'total_spent', 'retention_days', 'favorite_game_type', 'favorite_provider']].copy()

    # Identificar columnas categóricas por su índice
    categorical_indices = [data_for_clustering.columns.get_loc(col) for col in ['favorite_game_type', 'favorite_provider']]
    
    # Normalizar datos numéricos
    numerical_cols = ['age', 'total_spent', 'retention_days']
    scaler = StandardScaler()
    data_for_clustering[numerical_cols] = scaler.fit_transform(data_for_clustering[numerical_cols])
    
    matrix = data_for_clustering.to_numpy()

    k_range, costs = find_optimal_k(matrix, categorical_indices)

    # Gráfico del Codo
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=costs, mode='lines+markers'))
    fig_elbow.update_layout(title='Método del Codo para K-Prototypes',
                            xaxis_title='Número de Clústeres (k)',
                            yaxis_title='Costo (Disimilitud)')
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.info("El 'codo' (punto donde la curva se aplana) sugiere el número óptimo de clústeres. Un valor común es 3, 4 o 5 para este tipo de análisis.")

    st.subheader("2. Análisis de Clústeres")
    k_optimal = st.slider("Selecciona el número de clústeres (k) para analizar:", 2, 7, 4)

    # Entrenar modelo final y asignar clústeres
    clusters = k_prototypes_fit(matrix, categorical_indices, n_clusters=k_optimal).predict(matrix, categorical=categorical_indices)
    
    df_players['cluster'] = clusters
    data_for_clustering['cluster'] = clusters

    # Analizar los centroides
    cluster_analysis = df_players.groupby('cluster').agg(
        count=('player_id', 'size'),
        avg_age=('age', 'mean'),
        avg_total_spent=('total_spent', 'mean'),
        avg_retention_days=('retention_days', 'mean'),
        mode_game_type=('favorite_game_type', lambda x: x.mode()[0]),
        mode_provider=('favorite_provider', lambda x: x.mode()[0])
    ).reset_index()

    st.markdown(f"**Perfiles de los {k_optimal} cl煤steres encontrados:**")
    st.dataframe(cluster_analysis.style.format({
        'avg_age': '{:.1f}',
        'avg_total_spent': '${:,.2f}',
        'avg_retention_days': '{:.1f} d铆as'
    }))

    # Visualización de los clústeres
    fig_clusters = px.scatter(df_players, 
                              x='total_spent', 
                              y='age', 
                              color='cluster',
                              hover_data=['name', 'favorite_game_type', 'retention_days'],
                              title=f'Visualizaci贸n de {k_optimal} Cl煤steres de Jugadores',
                              labels={'total_spent': 'Gasto Total (USD)', 'age': 'Edad', 'cluster': 'Cl煤ster'})
    st.plotly_chart(fig_clusters, use_container_width=True)

# =================================================================================================
# PESTAÑA 2: APRIORI
# =================================================================================================
with tab2:
    st.header("Descubrimiento de Reglas de Asociación con Apriori")
    with st.expander("Justificación del Modelo", expanded=True):
        st.markdown("""
        Para descubrir combinaciones de comportamientos recurrentes, se necesita una técnica de reglas de asociación, y el algoritmo **Apriori** es el estándar de la industria para este fin. Este modelo es perfecto para analizar el "carrito de compras" de los juegos de un usuario.

        Nos permitirá extraer reglas de negocio valiosas como, por ejemplo:
        - *"El 70% de los usuarios que juegan 'Slots' también participan en 'Bingo'"* (confianza).
        - *"Los jugadores que prefieren 'Poker' y 'Blackjack' tienden a ser los que más gastan"* (asociación).

        El objetivo es identificar estas sinergias para potenciar estrategias de marketing cruzado.
        """)

    st.subheader("1. Preparación de Datos")
    st.markdown("Para Apriori, necesitamos transformar los datos a un formato transaccional, donde cada fila es un jugador y las columnas indican los juegos que ha jugado.")
    
    # Crear lista de transacciones (juegos por jugador)
    transactions = df_transactions.groupby('player_id')['game_type'].apply(list).values.tolist()
    
    # Usar TransactionEncoder para obtener una matriz one-hot
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_) # pyright: ignore[reportArgumentType]

    with st.expander("Ver datos en formato transaccional (One-Hot)"):
        st.dataframe(df_onehot.head())

    st.subheader("2. Entrenamiento y Validaci贸n")
    st.markdown("""
    El "entrenamiento" implica ejecutar el algoritmo para generar reglas. La "validaci贸n" se realiza ajustando las m茅tricas clave para filtrar solo las reglas m谩s relevantes.
    - **Soporte (Support):** Frecuencia con la que aparece un conjunto de juegos en todas las transacciones.
    - **Confianza (Confidence):** Probabilidad de que se juegue el juego B si ya se jug贸 el juego A.
    - **Lift:** Mide la fuerza de la asociaci贸n. Un lift > 1 indica una asociaci贸n positiva.
    """)

    col1, col2 = st.columns(2)
    min_support = col1.slider("Selecciona el Soporte M铆nimo (min_support):", 0.01, 0.2, 0.05)
    min_confidence = col2.slider("Selecciona la Confianza M铆nima (min_confidence):", 0.1, 0.8, 0.3)

    # Aplicar Apriori
    frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        st.warning("No se encontraron conjuntos de 铆tems frecuentes con el soporte actual. Intenta reducir el valor de 'Soporte M铆nimo'.")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        st.subheader("Reglas de Asociaci贸n Encontradas")
        if rules.empty:
            st.warning("No se encontraron reglas con la confianza actual. Intenta reducir el valor de 'Confianza M铆nima'.")
        else:
            # Limpiar y mostrar las reglas
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))


# =================================================================================================
# PESTA脩A 3: RANDOM FOREST
# =================================================================================================
with tab3:
    st.header("Predicci贸n de Comportamiento con Random Forest Regressor")
    with st.expander("Justificaci贸n del Modelo", expanded=True):
        st.markdown("""
        Para predecir el tiempo de retenci贸n y el gasto futuro, necesitamos modelos de regresi贸n. Se ha seleccionado **Random Forest** porque:
        - Es un modelo de *ensemble* robusto que **reduce el sobreajuste**.
        - Puede capturar **relaciones complejas y no lineales** entre las variables.
        - Proporciona la **importancia de las variables (feature importance)**, permitiendo identificar qu茅 factores son los predictores m谩s fuertes.
        """)

    st.subheader("1. Preparaci贸n de Datos y Divisi贸n")
    
    target_variable = st.selectbox(
        "Selecciona la variable a predecir:",
        ('D铆as de Retenci贸n (retention_days)', 'Gasto Futuro (future_spend)')
    )
    
    target_col = 'retention_days' if 'Retenci贸n' in target_variable else 'future_spend'
    
    # Preparar X e y
    features = ['age', 'total_spent', 'total_sessions', 'favorite_game_type', 'favorite_provider']
    X = df_players[features]
    y = df_players[target_col]

    # One-Hot Encoding para variables categ贸ricas
    X = pd.get_dummies(X, columns=['favorite_game_type', 'favorite_provider'], drop_first=True)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown(f"""
    - **Variable Objetivo (y):** `{target_col}`
    - **Variables Predictoras (X):** `{', '.join(features)}`
    - **Tama帽o del set de entrenamiento:** `{X_train.shape[0]} registros`
    - **Tama帽o del set de prueba:** `{X_test.shape[0]} registros`
    """)

    st.subheader("2. Entrenamiento y Evaluaci贸n del Modelo")
    
    # Entrenar el modelo
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # M茅tricas de evaluaci贸n
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    oob = rf_model.oob_score_

    st.markdown("Resultados en el conjunto de prueba:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Error Absoluto Medio (MAE)", f"{mae:.2f}")
    col2.metric("Error Cuadr谩tico Medio (MSE)", f"{mse:.2f}")
    col3.metric("Coeficiente de Determinaci贸n (R虏)", f"{r2:.2f}")
    col4.metric("Out-of-Bag (OOB) Score", f"{oob:.2f}", help="Puntuaci贸n de validaci贸n cruzada interna de Random Forest.")

    st.subheader("3. An谩lisis de Resultados")
    col1, col2 = st.columns([1, 1])

    with col1:
        # Gr谩fico de Predicciones vs. Valores Reales
        st.markdown("**Predicciones vs. Valores Reales**")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicciones'))
        fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                      mode='lines', name='L铆nea Ideal', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(title='Valores Reales vs. Predichos',
                               xaxis_title='Valores Reales',
                               yaxis_title='Valores Predichos')
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        # Gr谩fico de Importancia de Variables
        st.markdown("**Importancia de las Variables Predictoras**")
        importances = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
        importances = importances.sort_values('importance', ascending=True)
        
        fig_imp = px.bar(importances, 
                         x='importance', 
                         y='feature', 
                         orientation='h',
                         title='Importancia de cada variable en la predicci贸n')
        st.plotly_chart(fig_imp, use_container_width=True)