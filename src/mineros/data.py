import streamlit as st
from mineros import k_prototypes_fit

# Calcular costo para el método del codo
@st.cache_data
def find_optimal_k(data, categorical_indices):
    costs = []
    k_range = range(2, 8)
    for k in k_range:
        print(f"Training for {k} clusters...")
        kproto = k_prototypes_fit(data, categorical_indices, n_clusters=k)
        print(f"Cost for {k} clusters: {kproto.cost_}")
        costs.append(kproto.cost_)
    return k_range, costs