import streamlit as st
from kmodes.kprototypes import KPrototypes

@st.cache_resource
def k_prototypes_fit(data, categorical_indices, n_clusters):
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=0, random_state=42)
    kproto.fit(data, categorical=categorical_indices)
    return kproto