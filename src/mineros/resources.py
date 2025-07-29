import streamlit as st
from kmodes.kprototypes import KPrototypes
from sklearn.manifold import TSNE

@st.cache_resource
def k_prototypes_fit(data, categorical_indices, n_clusters):
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=0, random_state=42)
    kproto.fit(data, categorical=categorical_indices)
    return kproto

@st.cache_resource
def tsne_fit(data, categorical_cols):
    tsne_features_encoded = data.copy()
    for col in categorical_cols:
        if col in tsne_features_encoded.columns:
            tsne_features_encoded[col] = tsne_features_encoded[col].astype('category').cat.codes
    tsne = TSNE(n_components=2, random_state=42, perplexity=100)
    return tsne, tsne_features_encoded
