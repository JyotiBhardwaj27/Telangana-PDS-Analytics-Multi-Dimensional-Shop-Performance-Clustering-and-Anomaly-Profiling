import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from streamlit_folium import st_folium

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hdbscan

st.set_page_config(layout="wide")
st.title("Telangana PDS – Executive Behavioral Intelligence Dashboard")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("dashboard_dataset.csv")

master_df = load_data()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filters")

districts = sorted(master_df["distName"].dropna().unique())
years = sorted(master_df["year"].dropna().unique())

selected_district = st.sidebar.selectbox("District", ["All"] + districts)
selected_year = st.sidebar.selectbox("Year", ["All"] + list(years))

monitoring_level = st.sidebar.slider(
    "Monitoring Sensitivity Level",
    0.0, 1.0, 0.6, 0.05
)

df = master_df.copy()

if selected_district != "All":
    df = df[df["distName"] == selected_district]

if selected_year != "All":
    df = df[df["year"] == selected_year]

# =========================================================
# SHOP LEVEL AGGREGATION
# =========================================================
shop_df = df.groupby(
    ["year","distCode","shopNo","distName","latitude","longitude"]
)[
    ["utilization_ratio",
     "portability_ratio",
     "rice_wheat_ratio",
     "yearly_transaction_volatility"]
].mean().reset_index()

# Feature Engineering
shop_df["log_volatility"] = np.log1p(shop_df["yearly_transaction_volatility"])
shop_df["log_rice_wheat"] = np.log1p(shop_df["rice_wheat_ratio"])

features = [
    "utilization_ratio",
    "portability_ratio",
    "log_volatility",
    "log_rice_wheat"
]

X = shop_df[features].copy()
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# KMEANS CLUSTERING
# =========================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
shop_df["cluster"] = kmeans.fit_predict(X_scaled)

persona_map = {
    0: "Stable Rural Mainstream",
    1: "Urban Mobility-Driven",
    2: "Low-Variability Controlled",
    3: "High-Portability Transit Hubs"
}

shop_df["persona"] = shop_df["cluster"].map(persona_map)

# =========================================================
# HDBSCAN
# =========================================================
clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
labels_hdb = clusterer.fit_predict(X_scaled)
shop_df["hdb_label"] = labels_hdb

hdb_profile = (
    shop_df[shop_df["hdb_label"] != -1]
    .groupby("hdb_label")[features]
    .mean()
    .reset_index()
)

hdb_persona_map = {}

if not hdb_profile.empty:
    max_port = hdb_profile.loc[hdb_profile["portability_ratio"].idxmax(), "hdb_label"]
    min_port = hdb_profile.loc[hdb_profile["portability_ratio"].idxmin(), "hdb_label"]
    max_vol = hdb_profile.loc[hdb_profile["log_volatility"].idxmax(), "hdb_label"]

    hdb_persona_map[max_port] = "Transit Mobility Core"
    hdb_persona_map[min_port] = "Stable Core Shops"
    hdb_persona_map[max_vol] = "Urban Activity Core"

shop_df["hdb_persona"] = shop_df["hdb_label"].map(hdb_persona_map)
shop_df["hdb_persona"] = shop_df["hdb_persona"].fillna("Noise / Anomaly")

# =========================================================
# BEHAVIORAL INTENSITY INDEX
# =========================================================
shop_df["behavioral_intensity_index"] = (
    0.4 * shop_df["portability_ratio"] +
    0.3 * shop_df["utilization_ratio"] +
    0.3 * (shop_df["log_volatility"] / shop_df["log_volatility"].max())
)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🗺 Geospatial Intelligence",
    "📈 Cluster Analytics",
    "🔎 Shop Deep Dive",
    "⚠️ Anomaly & Risk Intelligence"
])

# =========================================================
# TAB 1 – EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Shops", len(shop_df))
    col2.metric("Avg Utilization", round(shop_df["utilization_ratio"].mean(),3))
    col3.metric("Avg Portability", round(shop_df["portability_ratio"].mean(),3))
    col4.metric("High Monitoring Priority %",
                round((shop_df["behavioral_intensity_index"] > monitoring_level).mean()*100,2))

    st.markdown("### Executive Insights")

    largest_persona = shop_df["persona"].value_counts().idxmax()
    noise_pct = round((shop_df["hdb_label"]==-1).mean()*100,2)

    st.info(f"""
    • Largest Behavioral Segment: **{largest_persona}**  
    • Monitoring Sensitivity Level: **{monitoring_level}**  
    • Behaviorally Distinct Shops (HDBSCAN): **{noise_pct}%**
    """)

    fig = px.pie(shop_df, names="persona", title="Persona Distribution")
    st.plotly_chart(fig, use_container_width=True)

    district_intensity = shop_df.groupby("distName")[
        "behavioral_intensity_index"
    ].mean().sort_values(ascending=False)

    st.markdown("### District Behavioral Intensity Ranking")
    st.dataframe(district_intensity.reset_index(), use_container_width=True)

    st.markdown("### Top 10 High Behavioral Intensity Shops")
    st.dataframe(
        shop_df.sort_values("behavioral_intensity_index", ascending=False)
        .head(10)[["year","shopNo","distName","behavioral_intensity_index","persona"]],
        use_container_width=True
    )

# =========================================================
# TAB 2 – MAP
# =========================================================
with tabs[1]:

    show_cluster = st.multiselect(
        "Select Personas",
        options=shop_df["persona"].unique(),
        default=list(shop_df["persona"].unique())
    )

    filtered_map_df = shop_df[
        shop_df["persona"].isin(show_cluster)
    ]

    if len(filtered_map_df) > 0:
        m = folium.Map(
            location=[filtered_map_df["latitude"].mean(),
                      filtered_map_df["longitude"].mean()],
            zoom_start=7
        )

        color_map = {
            "Stable Rural Mainstream": "blue",
            "Urban Mobility-Driven": "red",
            "Low-Variability Controlled": "green",
            "High-Portability Transit Hubs": "purple"
        }

        for _, row in filtered_map_df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=color_map.get(row["persona"], "gray"),
                fill=True,
                fill_opacity=0.7,
                popup=f"""
                Year: {row['year']} <br>
                Shop: {row['shopNo']} <br>
                Persona: {row['persona']} <br>
                Utilization: {round(row['utilization_ratio'],2)} <br>
                Portability: {round(row['portability_ratio'],2)}
                """
            ).add_to(m)

        st_folium(m, width=1200, height=550)

# =========================================================
# TAB 3 – CLUSTER ANALYTICS
# =========================================================
with tabs[2]:

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "Cluster": shop_df["persona"],
        "Anomaly": shop_df["hdb_persona"]
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.6,
        title="PCA Cluster Separation"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 – SHOP DEEP DIVE
# =========================================================
with tabs[3]:

    shop_input = st.number_input("Enter Shop Number", step=1)

    if shop_input:
        shop_data = shop_df[shop_df["shopNo"] == shop_input]

        if not shop_data.empty:
            row = shop_data.iloc[0]
            st.success(f"KMeans Persona: {row['persona']}")
            st.info(f"HDBSCAN Persona: {row['hdb_persona']}")

# =========================================================
# TAB 5 – ANOMALY INTELLIGENCE
# =========================================================
with tabs[4]:

    noise_count = (shop_df["hdb_label"] == -1).sum()
    st.metric("Anomalous Shops (HDBSCAN)", noise_count)

    st.dataframe(
        shop_df[shop_df["hdb_label"] == -1].head(20),
        use_container_width=True
    )
