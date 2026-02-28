# Telangana PDS Analytics  
## Multi-Dimensional Shop Performance Clustering & Behavioral Profiling

---

## 📖 Project Overview

This project performs multi-dimensional behavioral analysis of Fair Price Shops (FPS) under the Telangana Public Distribution System (PDS).

Using clustering and density-based anomaly detection techniques, the system:

- Segments shops into behavioral personas
- Identifies monitoring-priority shops
- Detects behaviorally distinct shops
- Provides district-level intelligence
- Enables executive decision support via interactive dashboard

---

## 🎯 Business Objective

The objective of this project is to apply unsupervised machine learning techniques to:

- Understand operational behavior of PDS shops
- Identify high behavioral intensity patterns
- Detect anomalous shop behavior
- Support monitoring prioritization
- Provide district-level analytical insights

---

## 🗂 Dataset

Source: Telangana Government Open Data Portal

Data Included:
- Shop transaction data (2023–2025)
- Card portability metrics
- Commodity distribution information
- Geospatial coordinates (Latitude & Longitude)

Data was consolidated into a shop-year analytical dataset.

---

## ⚙️ Methodology

### 1️⃣ Data Acquisition & Consolidation
- Merged multi-year datasets
- Performed joins on `shopNo` and `distCode`
- Created unified master dataset

### 2️⃣ Feature Engineering
Engineered behavioral indicators:
- Utilization Ratio
- Portability Ratio
- Rice-Wheat Intensity
- Log Transaction Volatility
- Behavioral Intensity Index (composite monitoring metric)

### 3️⃣ Clustering & Modeling
- **KMeans** → Generated 4 Behavioral Personas
- **HDBSCAN** → Density-based anomaly detection
- **PCA** → Dimensionality reduction for visualization
- **Silhouette Score** → Cluster validation

### 4️⃣ Behavioral Personas
- Stable Rural Mainstream
- Urban Mobility-Driven
- Low-Variability Controlled
- High-Portability Transit Hubs
- Noise / Anomaly (HDBSCAN distinct shops)

---

## 📊 Dashboard Features

Built using **Streamlit**

- District & Year filters
- Monitoring Sensitivity Slider
- Geospatial Cluster Map (Folium)
- PCA Cluster Projection
- Radar Persona Comparison
- Shop-Level Deep Dive
- Anomaly Intelligence View
- Downloadable filtered dataset

---

## 📈 Key Insights

- Majority shops fall into stable rural behavioral segments
- Urban regions exhibit higher portability intensity
- Less than 2% shops show behaviorally distinct patterns
- District-level variation observed in behavioral intensity

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- HDBSCAN
- Plotly
- Folium
- Streamlit

---

## 🚀 Deployment

Run locally:

```bash
streamlit run app.py
```

---

## 📌 Project Type

Unsupervised Learning | Clustering | Public Policy Analytics | Geospatial Intelligence | Executive Dashboarding

---

## 👤 Author

Jyoti Bharadwaj  
B.Tech (ECE) | Data Analytics & Machine Learning Enthusiast
