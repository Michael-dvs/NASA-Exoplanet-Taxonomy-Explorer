import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Exoplanet Taxonomy Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS: MODERN SPACE THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .cluster-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD & PREPARE DATA ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("all_exoplanets_2021.csv")
    selected_features = ['Orbital Period Days', 'Orbit Semi-Major Axis', 'Mass']
    
    # Ambil metadata untuk tooltip visualisasi
    meta_cols = ['Planet Name', 'Discovery Method', 'Discovery Year']
    df_clean = df[selected_features + meta_cols].dropna().copy()
    
    # Transformasi Logaritmik & Scaling
    df_log = np.log1p(df_clean[selected_features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log)
    
    # Training Model K-Means (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    # Mapping nama taksonomi astrofisika berdasarkan temuan model
    cluster_names = {
        0: 'Hot Jupiters',
        1: 'Cold Gas Giants',
        2: 'Terrestrial & Super-Earths'
    }
    df_clean['Cluster_Name'] = df_clean['Cluster_ID'].map(cluster_names)
    
    # Metrik Evaluasi
    metrics = {
        'silhouette': silhouette_score(X_scaled, df_clean['Cluster_ID']),
        'davies_bouldin': davies_bouldin_score(X_scaled, df_clean['Cluster_ID']),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, df_clean['Cluster_ID'])
    }
    
    return df_clean, scaler, kmeans, metrics, selected_features

# Inisialisasi Data
try:
    df_clean, scaler, kmeans_model, metrics, selected_features = load_and_preprocess_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: Pastikan file 'all_exoplanets_2021.csv' berada di direktori yang sama. Error: {e}")
    st.stop()

# --- SIDEBAR INTERAKTIF ---
with st.sidebar:
    st.title("🪐 Panel Filter")
    st.caption("Eksplorasi Taksonomi Eksoplanet NASA")
    
    cluster_options = df_clean['Cluster_Name'].unique().tolist()
    selected_clusters = st.multiselect(
        "Pilih Kategori Planet:",
        options=cluster_options,
        default=cluster_options
    )
    
    year_range = st.slider(
        "Tahun Penemuan:",
        int(df_clean['Discovery Year'].min()),
        int(df_clean['Discovery Year'].max()),
        (1995, int(df_clean['Discovery Year'].max()))
    )
    
    filtered_df = df_clean[
        (df_clean['Cluster_Name'].isin(selected_clusters)) &
        (df_clean['Discovery Year'].between(year_range[0], year_range[1]))
    ]
    
    st.divider()
    st.markdown("### 🧪 Uji Karakteristik Planet Baru")
    with st.form("predict_form"):
        st.caption("Masukkan spesifikasi planet untuk memprediksi klasternya:")
        inp_period = st.number_input("Orbital Period (Hari)", min_value=0.01, value=3.98, step=0.5)
        inp_axis = st.number_input("Orbit Semi-Major Axis (AU)", min_value=0.001, value=0.05, step=0.01)
        inp_mass = st.number_input("Massa (Earth Mass)", min_value=0.01, value=310.0, step=10.0)
        
        submitted = st.form_submit_button("🔍 Prediksi Kategori")
        if submitted:
            user_input = np.log1p([[inp_period, inp_axis, inp_mass]])
            scaled_input = scaler.transform(user_input)
            pred_cluster = kmeans_model.predict(scaled_input)[0]
            names_map = {0: 'Hot Jupiters 🔥', 1: 'Cold Gas Giants ❄️', 2: 'Terrestrial & Super-Earths 🌍'}
            st.success(f"Hasil Klasifikasi: **{names_map[pred_cluster]}**")

# --- HEADER SECTION ---
st.title("🌌 NASA Exoplanet Taxonomy Explorer")
st.markdown("""
Aplikasi web ini menggunakan algoritma **Unsupervised Machine Learning (K-Means Clustering)** untuk memetakan dan mengelompokkan 
ribuan planet di luar tata surya secara otomatis berdasarkan parameter orbit dan massa fisisnya.
""")

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Eksoplanet Teranalisis", f"{len(filtered_df):,} Data")
col2.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
col3.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin']:.4f}")
col4.metric("Calinski-Harabasz Index", f"{metrics['calinski_harabasz']:.1f}")

st.write("")

# --- VISUALIZATION TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 3D Orbital Feature Space", "📊 2D Scatter & Klasterisasi", "📋 Profil Statistik Klaster"])

color_map = {
    'Hot Jupiters': '#EF553B',
    'Cold Gas Giants': '#636EFA',
    'Terrestrial & Super-Earths': '#00CC96'
}

with tab1:
    st.subheader("Ruang Fitur 3D Interaktif (Log-Scaled)")
    st.caption("Visualisasi 3 dimensi memetakan kombinasi Massa, Jarak Orbit (AU), dan Periode Orbit (Hari).")
    
    fig_3d = px.scatter_3d(
        filtered_df,
        x=np.log10(filtered_df['Orbital Period Days']),
        y=np.log10(filtered_df['Orbit Semi-Major Axis']),
        z=np.log10(filtered_df['Mass']),
        color='Cluster_Name',
        hover_name='Planet Name',
        hover_data={'Discovery Method': True, 'Discovery Year': True},
        color_discrete_map=color_map,
        labels={
            'x': 'Log10(Orbital Period [Hari])',
            'y': 'Log10(Semi-Major Axis [AU])',
            'z': 'Log10(Mass [Massa Bumi])'
        },
        opacity=0.8,
        height=620
    )
    fig_3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        ),
        margin=dict(l=0, r=0, b=0, t=10)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Distribusi Massa vs Periode Orbit")
    fig_2d = px.scatter(
        filtered_df,
        x='Orbital Period Days',
        y='Mass',
        color='Cluster_Name',
        color_discrete_map=color_map,
        hover_name='Planet Name',
        log_x=True,
        log_y=True,
        labels={'Orbital Period Days': 'Periode Orbit (Hari) [Log Scale]', 'Mass': 'Massa Planet (Massa Bumi) [Log Scale]'},
        template="plotly_dark",
        height=520
    )
    fig_2d.update_traces(marker=dict(size=7, opacity=0.75))
    st.plotly_chart(fig_2d, use_container_width=True)

with tab3:
    st.subheader("Rangkuman Karakteristik Tiap Klaster")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div class="cluster-card">
            <h4 style="color: #EF553B;">🔥 Hot Jupiters (Klaster 0)</h4>
            <p><b>Ciri Utama:</b> Massa masif (~309 M⊕) yang berada sangat dekat dengan bintang induknya.</p>
            <ul>
                <li><b>Median Periode:</b> ~3.98 hari</li>
                <li><b>Median Jarak:</b> ~0.05 AU</li>
                <li><b>Kondisi:</b> Ekstrem panas, atmosfer terbakar radiasi bintang.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown("""
        <div class="cluster-card">
            <h4 style="color: #636EFA;">❄️ Cold Gas Giants (Klaster 1)</h4>
            <p><b>Ciri Utama:</b> Raksasa gas paling masif (~772 M⊕) yang berada di orbit luar yang dingin.</p>
            <ul>
                <li><b>Median Periode:</b> ~918 hari (~2.5 tahun)</li>
                <li><b>Median Jarak:</b> ~1.97 AU</li>
                <li><b>Kondisi:</b> Mirip Jupiter dan Saturnus di tata surya kita.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[2]:
        st.markdown("""
        <div class="cluster-card">
            <h4 style="color: #00CC96;">🌍 Terrestrial & Super-Earths (Klaster 2)</h4>
            <p><b>Ciri Utama:</b> Kelompok planet bermassa rendah (~8.8 M⊕) dengan karakteristik mirip Bumi.</p>
            <ul>
                <li><b>Median Periode:</b> ~13.22 hari</li>
                <li><b>Median Jarak:</b> ~0.096 AU</li>
                <li><b>Kondisi:</b> Target riset utama pencarian zona laik huni (Habitable Zone).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("#### Data Agregasi Median dan Rata-Rata")
    summary_table = df_clean.groupby('Cluster_Name')[selected_features].agg(['median', 'mean']).round(2)
    st.dataframe(summary_table, use_container_width=True)