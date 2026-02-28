import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from streamlit_folium import st_folium
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("Telangana PDS – Executive Behavioral Intelligence Dashboard")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("dashboard_dataset.csv")

df = load_data()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filters")

districts = sorted(df["distName"].dropna().unique())
years = sorted(df["year"].dropna().unique())

selected_district = st.sidebar.selectbox("District", ["All"] + districts)
selected_year = st.sidebar.selectbox("Year", ["All"] + list(years))

monitoring_level = st.sidebar.slider(
    "Monitoring Sensitivity Level",
    0.0, 1.0, 0.6, 0.05
)

# Apply filters
if selected_district != "All":
    df = df[df["distName"] == selected_district]

if selected_year != "All":
    df = df[df["year"] == selected_year]

shop_df = df.copy()

features = [
    "utilization_ratio",
    "portability_ratio",
    "log_volatility",
    "log_rice_wheat"
]

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🗺 Geospatial Intelligence",
    "📈 Cluster Analytics",
    "🔎 Shop Deep Dive",
    "⚠️ Anomaly Intelligence"
])

# =========================================================
# TAB 1 – EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Shops", len(shop_df))
    col2.metric("Avg Utilization", round(shop_df["utilization_ratio"].mean(), 3))
    col3.metric("Avg Portability", round(shop_df["portability_ratio"].mean(), 3))
    col4.metric(
        "High Monitoring Priority %",
        round((shop_df["behavioral_intensity_index"] > monitoring_level).mean()*100, 2)
    )

    st.markdown("### Executive Insights")

    largest_persona = shop_df["persona"].value_counts().idxmax()
    anomaly_pct = round((shop_df["hdb_persona"] == "Noise / Anomaly").mean()*100, 2)

    st.info(f"""
    • Largest Behavioral Segment: **{largest_persona}**  
    • Monitoring Sensitivity Level: **{monitoring_level}**  
    • Behaviorally Distinct Shops: **{anomaly_pct}%**
    """)

    # Persona Distribution
    fig_pie = px.pie(
        shop_df,
        names="persona",
        title="Persona Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # District Ranking
    st.markdown("### District Behavioral Intensity Ranking")

    district_intensity = shop_df.groupby("distName")[
        "behavioral_intensity_index"
    ].mean().sort_values(ascending=False)

    st.dataframe(
        district_intensity.reset_index(),
        use_container_width=True
    )

    # Top Shops
    st.markdown("### Top 10 High Behavioral Intensity Shops")

    st.dataframe(
        shop_df.sort_values("behavioral_intensity_index", ascending=False)
        .head(10)[
            ["year","shopNo","distName","behavioral_intensity_index","persona"]
        ],
        use_container_width=True
    )

# =========================================================
# TAB 2 – MAP
# =========================================================
with tabs[1]:

    st.subheader("Interactive Cluster Map")

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
    else:
        st.warning("No data for selected filters.")

# =========================================================
# TAB 3 – CLUSTER ANALYTICS
# =========================================================
with tabs[2]:

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(shop_df[features])

    pca_df = pd.DataFrame({
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "Persona": shop_df["persona"],
        "Anomaly": shop_df["hdb_persona"]
    })

    fig_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Persona",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="PCA Cluster Separation"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Radar Chart
    cluster_profile = shop_df.groupby("persona")[features].mean()

    normalized = (cluster_profile - cluster_profile.min()) / \
                 (cluster_profile.max() - cluster_profile.min())

    fig_radar = go.Figure()

    for persona in normalized.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized.loc[persona].values,
            theta=features,
            fill='toself',
            name=persona
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="Normalized Persona Behavioral Radar"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

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

            cluster_avg = shop_df[
                shop_df["persona"] == row["persona"]
            ][features].mean()

            deviation = row[features] - cluster_avg

            compare_df = pd.DataFrame({
                "Metric": features,
                "Shop Value": row[features].values,
                "Cluster Avg": cluster_avg.values,
                "Deviation": deviation.values
            })

            st.dataframe(compare_df, use_container_width=True)

        else:
            st.warning("Shop not found.")

# =========================================================
# TAB 5 – ANOMALY INTELLIGENCE
# =========================================================
with tabs[4]:

    anomaly_count = (shop_df["hdb_persona"] == "Noise / Anomaly").sum()

    st.metric("Behaviorally Distinct Shops", anomaly_count)

    st.write("Sample of Behaviorally Distinct Shops:")

    st.dataframe(
        shop_df[shop_df["hdb_persona"] == "Noise / Anomaly"]
        .head(20),
        use_container_width=True
    )

    st.download_button(
        "Download Filtered Dataset",
        shop_df.to_csv(index=False),
        file_name="telangana_pds_filtered.csv",
        mime="text/csv"
    )
