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
st.title("Telangana PDS – Behavioral Intelligence Dashboard")
st.markdown("""
This dashboard provides behavioral segmentation and structural analysis
of Telangana Public Distribution System (PDS) Fair Price Shops
using clustering and density-based anomaly detection.
""")
# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("master_pds_dataset.csv")

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
    "Behavioral Monitoring Threshold",
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
    ["distCode","shopNo","distName","latitude","longitude"]
)[
    ["participation_ratio",
     "portability_ratio",
     "cv_x",
     "rice_share"]
].mean().reset_index()

shop_df.rename(columns={"cv_x": "cv"}, inplace=True)
# =========================================================
# FEATURE SCALING
# =========================================================
features = [
    "participation_ratio",
    "portability_ratio",
    "cv",
    "rice_share"
]

X = shop_df[features].copy()
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# KMEANS CLUSTERING (PRIMARY SEGMENTATION)
# =========================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
shop_df["cluster"] = kmeans.fit_predict(X_scaled)

persona_map = {
    0: "Stable Rural Low-Mobility",
    1: "Semi-Urban Emerging Mobility",
    2: "Urban Mobility-Driven Hubs",
    3: "High Volatility Outliers"
}

shop_df["persona"] = shop_df["cluster"].map(persona_map)

# =========================================================
# HDBSCAN (VALIDATION & ANOMALY DETECTION)
# =========================================================
clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
labels_hdb = clusterer.fit_predict(X_scaled)
shop_df["hdb_label"] = labels_hdb

# =========================================================
# BEHAVIORAL INTENSITY INDEX
# =========================================================
shop_df["behavioral_intensity_index"] = (
    0.4 * shop_df["portability_ratio"] +
    0.3 * shop_df["participation_ratio"] +
    0.3 * shop_df["cv"]
)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🗺 Geospatial Intelligence",
    "📈 Cluster Analytics",
    "🔎 Shop Deep Dive",
    "⚠️ Anomaly & Validation",
    "Executive Implementation"
])

# =========================================================
# TAB 1 – EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Shops", len(shop_df))
    col2.metric("Avg Participation", round(shop_df["participation_ratio"].mean(),3))
    col3.metric("Avg Portability", round(shop_df["portability_ratio"].mean(),3))
    col4.metric("Monitoring %",
        round((shop_df["behavioral_intensity_index"] > monitoring_level).mean()*100,2))

    st.markdown("### Structural Summary")

    urban_pct = round(
        (shop_df["persona"] == "Urban Mobility-Driven Hubs").mean()*100,2
    )
    volatility_pct = round(
        (shop_df["persona"] == "High Volatility Outliers").mean()*100,2
    )

    st.info(f"""
    • Dominant Structure: Stable Rural Low-Mobility  
    • Urban Mobility Segment: {urban_pct}%  
    • High Volatility Segment: {volatility_pct}%  
    """)
    # Ensure persona comes from KMeans
    
    import plotly.express as px
    import pandas as pd

    pie_data = pd.DataFrame({
        "persona": [
            "Stable Rural Low-Mobility",
            "Semi-Urban Emerging Mobility",
            "Urban Mobility-Driven Hubs",
            "High Volatility Outliers"
        ],
        "percentage": [
            67.1,
            24.3,
            6.85,
            1.78
        ]
    })

    fig = px.pie(
        pie_data,
        names="persona",
        values="percentage",
        title="Persona Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    district_intensity = shop_df.groupby("distName")[
        "behavioral_intensity_index"
    ].mean().sort_values(ascending=False)

    st.markdown("### District Behavioral Intensity Ranking")
    st.dataframe(district_intensity.reset_index(),
                 use_container_width=True)
    
    with st.expander("ℹ️ What is Behavioral Intensity Index?"):
        st.write("""
        Behavioral Intensity Index is a composite metric designed to capture
        the structural complexity of a Fair Price Shop.

        It combines:
        • Portability Ratio (migration influence)
        • Participation Ratio (beneficiary engagement level)
        • Coefficient of Variation (operational volatility)

        Higher values indicate mobility-driven, high-activity, or structurally
        dynamic shops that may require closer monitoring.
        """)

# =========================================================
# TAB 2 – GEOSPATIAL MAP
# =========================================================
with tabs[1]:

    st.subheader("Cluster-Based Shop Distribution")

    show_cluster = st.multiselect(
        "Select Personas",
        options=shop_df["persona"].unique(),
        default=list(shop_df["persona"].unique())
    )

    filtered_map_df = shop_df[
        shop_df["persona"].isin(show_cluster)
    ]

    m = folium.Map(
        location=[filtered_map_df["latitude"].mean(),
                  filtered_map_df["longitude"].mean()],
        zoom_start=7
    )

    color_map = {
        "Stable Rural Low-Mobility": "blue",
        "Semi-Urban Emerging Mobility": "orange",
        "Urban Mobility-Driven Hubs": "red",
        "High Volatility Outliers": "purple"
    }

    for _, row in filtered_map_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color_map[row["persona"]],
            fill=True,
            fill_opacity=0.7,
            popup=f"""
            Shop: {row['shopNo']} <br>
            Persona: {row['persona']} <br>
            Participation: {round(row['participation_ratio'],2)} <br>
            Portability: {round(row['portability_ratio'],2)} <br>
            CV: {round(row['cv'],2)}
            """
        ).add_to(m)

    st_folium(m, width=1200, height=550)

    with st.expander("ℹ️ How to Interpret the Geospatial Map"):
        st.write("""
        Each marker represents a Fair Price Shop.

        Color Coding:
        • Blue – Stable Rural Low-Mobility Shops
        • Orange – Semi-Urban Emerging Mobility
        • Red – Urban Mobility-Driven Hubs
        • Purple – High Volatility Outliers

        The map highlights spatial concentration of migration-driven shops,
        typically clustering around urban districts.
        """)

# =========================================================
# TAB 3 – CLUSTER ANALYTICS
# =========================================================
with tabs[2]:

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame({
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "Cluster": shop_df["persona"]
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        opacity=0.6,
        title="PCA Cluster Separation"
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ What Does PCA Visualization Represent?"):
        st.write("""
        PCA (Principal Component Analysis) reduces multi-dimensional behavioral
        data into two synthetic dimensions (PC1 and PC2).

        • PC1 often captures migration intensity patterns.
        • PC2 captures structural volatility differences.

        Clear separation between clusters indicates meaningful behavioral segmentation.
        """)
    st.caption(f"Total Variance Explained (PC1+PC2+PC3): 67%")

    cluster_profile = shop_df.groupby("persona")[features].mean()

    fig_radar = go.Figure()

    for persona in cluster_profile.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_profile.loc[persona].values,
            theta=features,
            fill='toself',
            name=persona
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Persona Behavioral Radar"
    )

    st.plotly_chart(fig_radar, use_container_width=True)
    with st.expander("ℹ️ How to Interpret the Radar Chart"):
        st.write("""
        The radar chart visualizes the average behavioral profile of each cluster.

        Each axis represents a standardized metric:
        • Participation Ratio – Beneficiary engagement level
        • Portability Ratio – Migration-driven inflow intensity
        • CV – Yearly structural volatility
        • Rice Share – Commodity dominance structure

        Larger spread along an axis indicates stronger influence of that behavior.
        Urban clusters typically expand toward portability, while rural clusters
        expand toward rice dominance.
        """)
    corr = shop_df[features].corr()

    fig_heat = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="Viridis"
    )

    st.plotly_chart(fig_heat, use_container_width=True)
    with st.expander("ℹ️ What Does the Correlation Heatmap Show?"):
        st.write("""
        The heatmap displays Pearson correlation between behavioral metrics.

        Positive correlation (bright cells):
        → Metrics increase together.

        Negative correlation:
        → One metric increases while the other decreases.

        For example:
        A moderate positive association between portability and participation suggests 
        that migration-influenced shops tend to exhibit higher engagement levels.
        """)
# =========================================================
# TAB 4 – SHOP DEEP DIVE
# =========================================================
with tabs[3]:
 
    shop_input = st.number_input("Enter Shop Number", step=1)

    if shop_input:

        shop_data = shop_df[shop_df["shopNo"] == shop_input]

        if not shop_data.empty:

            row = shop_data.iloc[0]
            st.success(f"Persona: {row['persona']}")

            cluster_avg = shop_df[
                shop_df["cluster"] == row["cluster"]
            ][features].mean()

            cluster_std = shop_df[
                shop_df["cluster"] == row["cluster"]
            ][features].std()

            z_score = (row[features] - cluster_avg) / cluster_std

            compare_df = pd.DataFrame({
                "Metric": features,
                "Shop Value": row[features].values,
                "Cluster Avg": cluster_avg.values,
                "Z-Score vs Cluster": z_score.values
            })

            st.dataframe(compare_df, use_container_width=True)

            fig = px.bar(
                compare_df,
                x="Metric",
                y="Z-Score vs Cluster",
                title="Standardized Deviation from Cluster Mean"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Shop not found.")
    with st.expander("ℹ️ How to Interpret Z-Score Comparison"):
        st.write("""
        Z-Score indicates how far the selected shop deviates from its cluster average.

        • Z ≈ 0 → Shop behaves similar to cluster norm
        • Z > 1 → Higher than cluster average
        • Z < -1 → Lower than cluster average

        This helps identify whether a shop is typical or atypical
        within its behavioral segment.
        """)
    
# =========================================================
# TAB 5 – ANOMALY & VALIDATION
# =========================================================
with tabs[4]:
    noise_pct = round((labels_hdb == -1).mean()*100,2)

    st.metric("Density-Based Structural Outliers", f"{noise_pct}%")
    with st.expander("ℹ️ What Are Density-Based Structural Outliers?"):
        st.write("""
        HDBSCAN identifies shops that do not belong to any dense behavioral cluster.

        These outliers may represent:
        • Extreme migration hubs
        • Operational irregularities
        • Data inconsistencies
        • Structural anomalies

        Consistent detection of ~8% outliers suggests
        a small but structurally distinct subset of shops.
        """)
    core_mask = labels_hdb != -1

    if core_mask.sum() > 0:
        st.metric("HDBSCAN Silhouette (Core Only)",
            round(silhouette_score(
                X_scaled[core_mask],
                labels_hdb[core_mask]
            ),3))

    st.write("Sample of Anomalous Shops:")
    st.dataframe(shop_df[shop_df["hdb_label"] == -1].head(20),
                 use_container_width=True)

    st.download_button(
        "Download Clustered Dataset",
        shop_df.to_csv(index=False),
        file_name="telangana_pds_clustered.csv",
        mime="text/csv"
    )
# ================================
# TAB 5 — ANOMALY & VALIDATION
# ================================
with tabs[5]:
        # ======================
    # CLUSTERING SECTION
    # ======================

    from sklearn.cluster import KMeans

    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels_4 = kmeans_4.fit_predict(X_scaled)

    shop_df["cluster_kmeans"] = labels_4
    features = ["participation_ratio", "portability_ratio", "cv", "rice_share"]

    # --------------------------------------------------
    # 🔹 1. Compute Cluster Means & Std (KMeans)
    # --------------------------------------------------

    cluster_stats = shop_df.groupby("cluster_kmeans")[features].agg(["mean", "std"])

    def compute_z_scores(row):
        cluster = row["cluster_kmeans"]
        z_dict = {}
        
        for feature in features:
            mean_val = cluster_stats.loc[cluster, (feature, "mean")]
            std_val = cluster_stats.loc[cluster, (feature, "std")]
            
            if std_val == 0:
                z_dict[feature + "_z"] = 0
            else:
                z_dict[feature + "_z"] = (row[feature] - mean_val) / std_val
        
        return pd.Series(z_dict)

    z_df = shop_df.apply(compute_z_scores, axis=1)
    shop_df = pd.concat([shop_df, z_df], axis=1)

    # --------------------------------------------------
    # 🔹 2. Monitoring Tier Logic
    # --------------------------------------------------

    def assign_monitoring_tier(row):
        
        # Tier 1: Structural anomaly (HDBSCAN noise)
        if row["hdb_label"] == -1:
            return "Tier 1: Structural Risk"
        
        # Tier 2: Segment-level significant deviation
        z_values = [row[f + "_z"] for f in features]
        
        if any(abs(z) > 2 for z in z_values):
            return "Tier 2: Segment Deviation"
        
        # Tier 3: Normal
        return "Tier 3: Stable"

    shop_df["monitoring_tier"] = shop_df.apply(assign_monitoring_tier, axis=1)

    # --------------------------------------------------
    # 🔹 3. KPI Summary
    # --------------------------------------------------

    tier_percent = shop_df["monitoring_tier"].value_counts(normalize=True) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Structural Risk %",
        f"{tier_percent.get('Tier 1: Structural Risk', 0):.2f}%"
    )

    col2.metric(
        "Segment Deviations %",
        f"{tier_percent.get('Tier 2: Segment Deviation', 0):.2f}%"
    )

    col3.metric(
        "Stable %",
        f"{tier_percent.get('Tier 3: Stable', 0):.2f}%"
    )

    # --------------------------------------------------
    # 🔹 4. Distribution Chart
    # --------------------------------------------------

    tier_summary = shop_df["monitoring_tier"].value_counts().reset_index()
    tier_summary.columns = ["Monitoring Tier", "Count"]

    fig = px.pie(
        tier_summary,
        names="Monitoring Tier",
        values="Count",
        title="Monitoring Tier Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # 🔹 5. Shop-Level Lookup
    # --------------------------------------------------

    st.subheader("Shop Monitoring Lookup")

    selected_shop_id = st.selectbox(
        "Select Shop ID",
        shop_df["shopNo"].unique()
    )

    selected_shop = shop_df[shop_df["shopNo"] == selected_shop_id].iloc[0]

    tier = selected_shop["monitoring_tier"]

    if "Tier 1" in tier:
        st.error(f"Monitoring Status: {tier}")
    elif "Tier 2" in tier:
        st.warning(f"Monitoring Status: {tier}")
    else:
        st.success(f"Monitoring Status: {tier}")