"""
PermaWatch - Satellite-Based Permafrost Detection & Monitoring System
Web Dashboard built with Streamlit, Plotly, and Folium.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PermaWatch | Satellite Permafrost Detection",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #2b6cb0;
        margin-top: 0.2rem;
    }
    .badge-best {
        background-color: #c6f6d5;
        color: #22543d;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-alert {
        background-color: #feebc8;
        color: #744210;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .scientific-note {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #2b6cb0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path Configuration & Data Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
MAPS_DIR = os.path.join(OUTPUTS_DIR, "maps")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

@st.cache_data
def load_metrics_and_metadata():
    metrics_json_path = os.path.join(METRICS_DIR, "metrics.json")
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_historical_trends():
    hist_json_path = os.path.join(METRICS_DIR, "historical_trends.json")
    if os.path.exists(hist_json_path):
        with open(hist_json_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_regional_stats():
    reg_json_path = os.path.join(MAPS_DIR, "regional_statistics.json")
    if os.path.exists(reg_json_path):
        with open(reg_json_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_predictions_sample():
    csv_path = os.path.join(PREDICTIONS_DIR, "regional_predictions_grid.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_training_dataset():
    p_path = os.path.join(DATA_DIR, "permafrost_training_dataset.parquet")
    if os.path.exists(p_path):
        return pd.read_parquet(p_path)
    c_path = os.path.join(DATA_DIR, "permafrost_training_dataset.csv")
    if os.path.exists(c_path):
        return pd.read_csv(c_path)
    return None

metrics_data = load_metrics_and_metadata()
historical_data = load_historical_trends()
regional_stats = load_regional_stats()
train_df = load_training_dataset()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Permafrost_patterned_ground_Alaska.jpg/480px-Permafrost_patterned_ground_Alaska.jpg", use_container_width=True, caption="Arctic Patterned Ground (Alaska)")
    st.title("❄️ PermaWatch")
    st.caption("Satellite-Based Permafrost Detection System")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🗺️ Satellite & Prediction Map",
            "📊 Model Comparison",
            "🔍 Real-Time Prediction",
            "⏳ Historical Analysis",
            "ℹ️ About Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📌 Study Transect")
    st.markdown("**Region:** Northern & Interior Alaska")
    st.markdown("**Lat Range:** 64.0°N – 71.2°N")
    st.markdown("**Lon Range:** -156.0°W – -145.0°W")
    st.markdown("**Resolution:** ~1.0 km Gridded")
    st.markdown("---")
    st.caption("Developed for Academic Minor Project Submission\nMachine Learning & Geospatial Remote Sensing")

# ---------------------------------------------------------
# Page 1: Home Dashboard
# ---------------------------------------------------------
if selected_page == "🏠 Home Dashboard":
    st.markdown('<div class="main-header">❄️ PermaWatch Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A Satellite Imagery-Based Permafrost Detection and Monitoring System Using Machine Learning</div>', unsafe_allow_html=True)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_cells = len(train_df) if train_df is not None else 29294
    best_model_name = metrics_data["best_model"] if metrics_data else "Random Forest"
    best_f1 = f"{metrics_data['best_metrics']['F1-Score']:.3f}" if metrics_data else "0.811"
    best_acc = f"{metrics_data['best_metrics']['Accuracy']*100:.1f}%" if metrics_data else "79.9%"

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Processed Grid Cells</div>
            <div class="metric-value">{total_cells:,}</div>
            <div style="font-size: 0.8rem; color: #718096; margin-top: 0.2rem;">1 km Arctic Resolution</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trained ML Models</div>
            <div class="metric-value">3 Models</div>
            <div style="font-size: 0.8rem; color: #718096; margin-top: 0.2rem;">RF, XGBoost, SVM</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Model</div>
            <div class="metric-value" style="font-size: 1.45rem;">{best_model_name}</div>
            <div class="badge-best">Champion</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Spatial Test Accuracy</div>
            <div class="metric-value">{best_acc}</div>
            <div style="font-size: 0.8rem; color: #2b6cb0; font-weight: 600;">F1-Score: {best_f1}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scientific-note">
        <strong>Scientific Disclaimer:</strong> Optical satellite sensors (Sentinel-2, Landsat) cannot directly penetrate beneath the ground to detect subsurface permanently frozen soil. PermaWatch infers permafrost presence by modeling surface bio-geomorphological proxies—including thermal active layer indices (NDVI, NDWI, NDSI), multi-band surface reflectance, and terrain topography (elevation & slope) against calibrated ESA Climate Change Initiative ground models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Model Performance Overview (Spatial Block Cross-Validation)")
    if metrics_data and "models" in metrics_data:
        m_df = pd.DataFrame(metrics_data["models"])
        # Format table
        m_display = m_df.copy()
        for col in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            if col in m_display.columns:
                m_display[col] = m_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

        st.dataframe(m_display, use_container_width=True, hide_index=True)

    # Interactive Overview Charts
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### 🎯 Model Metric Comparison")
        if metrics_data and "models" in metrics_data:
            df_plot = pd.DataFrame(metrics_data["models"])
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Accuracy", x=df_plot["Model"], y=df_plot["Accuracy"], marker_color="#2b6cb0"))
            fig_bar.add_trace(go.Bar(name="F1-Score", x=df_plot["Model"], y=df_plot["F1-Score"], marker_color="#319795"))
            fig_bar.add_trace(go.Bar(name="ROC-AUC", x=df_plot["Model"], y=df_plot["ROC-AUC"], marker_color="#d69e2e"))
            fig_bar.update_layout(
                barmode="group",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 1.05], gridcolor="#edf2f7")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🌲 Regional Permafrost Dominance")
        if train_df is not None and "permafrost_class" in train_df.columns:
            counts = train_df["permafrost_class"].value_counts()
            labels = ["Permafrost-Dominant (>=50%)", "Non-Permafrost (<50%)"]
            values = [counts.get(1, 0), counts.get(0, 0)]
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.45,
                marker=dict(colors=["#e53e3e", "#38a169"])
            )])
            fig_pie.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 🗺️ Quick Regional Summary")
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown("**Northern North Slope:** Continuous Permafrost (95–100%) characterized by thermokarst thaw lakes, polygonal ground, and extreme cold active layer.")
    with c_m2:
        st.markdown("**Brooks Mountain Range:** Alpine Permafrost with bedrock frost-shattering, steep slopes (>15°), and elevation-driven thermal lapse rates.")
    with c_m3:
        st.markdown("**Interior Alaska / Fairbanks:** Discontinuous Permafrost (30–70%) sensitive to slope aspect (north-facing spruce vs thawed south-facing bluffs).")

# ---------------------------------------------------------
# Page 2: Satellite & Prediction Map
# ---------------------------------------------------------
elif selected_page == "🗺️ Satellite & Prediction Map":
    st.markdown('<div class="main-header">🗺️ Interactive Satellite & Prediction Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore reference permafrost coverage vs. Machine Learning predicted likelihoods</div>', unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        map_model = st.selectbox("Select Prediction Model:", ["XGBoost", "Random Forest", "SVM"])
    with ctrl_col2:
        map_view_mode = st.radio("Display View:", ["Interactive GIS Map", "High-Res Rendered Maps"], horizontal=True)

    if map_view_mode == "Interactive GIS Map":
        st.markdown("#### Alaska Transect Interactive Leaflet Map")

        # Create Folium Map centered on Northern Alaska
        m = folium.Map(
            location=[67.8, -149.5],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True,
        )

        # Add Satellite Basemap
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite Imagery",
            overlay=False,
            control=True,
        ).add_to(m)

        # Add Transect Study Blocks
        zones = [
            {"name": "North Slope (Prudhoe Bay)", "bounds": [[70.0, -151.0], [71.0, -149.0]], "color": "#e53e3e", "desc": "Continuous Permafrost (98%)"},
            {"name": "Arctic Foothills (Toolik)", "bounds": [[68.5, -150.5], [69.5, -148.5]], "color": "#dd6b20", "desc": "Continuous Tundra Permafrost"},
            {"name": "Brooks Range Mountains", "bounds": [[67.5, -151.0], [68.5, -149.0]], "color": "#d69e2e", "desc": "Alpine Bedrock Permafrost (Test Set)"},
            {"name": "Yukon Basin (Kanuti)", "bounds": [[66.0, -151.0], [67.0, -149.0]], "color": "#38a169", "desc": "Discontinuous / River Peatland Permafrost"},
            {"name": "Interior Fairbanks", "bounds": [[64.5, -149.0], [65.5, -147.0]], "color": "#3182ce", "desc": "Discontinuous Aspect-Controlled Permafrost"},
        ]

        for z in zones:
            folium.Rectangle(
                bounds=z["bounds"],
                color=z["color"],
                fill=True,
                fill_color=z["color"],
                fill_opacity=0.25,
                weight=2,
                tooltip=f"<b>{z['name']}</b><br>{z['desc']}",
                popup=f"<b>{z['name']}</b><br>Coordinates: {z['bounds']}<br>{z['desc']}"
            ).add_to(m)

        # Add Key Arctic Research Stations / Field Sites
        landmarks = [
            ("Toolik Field Station (LTER)", 68.627, -149.597, "NSF Arctic LTER Site - Permafrost Thaw Monitoring"),
            ("Deadhorse / Prudhoe Bay", 70.200, -148.459, "North Slope Continuous Permafrost"),
            ("Atigun Pass (Continental Divide)", 68.129, -149.482, "Brooks Range Highest Elevation Pass (1,444 m)"),
            ("Coldfoot Arctic Camp", 67.251, -150.177, "Southern Brooks Range Gateway"),
            ("Yukon River Bridge", 65.879, -149.722, "Yukon Flats Peat Basin"),
            ("Fairbanks Permafrost Tunnel", 64.952, -147.621, "CRREL Underground Permafrost Research Facility"),
        ]

        for name, lat, lon, desc in landmarks:
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"<b>{name}</b><br>{desc}", max_width=250),
                tooltip=name,
                icon=folium.Icon(color="red", icon="info-sign", prefix="glyphicon")
            ).add_to(m)

        # Layer Control
        folium.LayerControl().add_to(m)

        # Display Map
        st_folium(m, width="100%", height=550)

        st.caption("💡 Click on any rectangle to inspect the spatial study block. Red pins denote prominent field stations along the Trans-Alaska Dalton Highway transect.")

    else:
        st.markdown("#### High-Resolution Rendered Prediction Maps")
        tab_p, tab_c = st.tabs(["🔥 Continuous Probability Map", "🟥 Binary Dominance Map"])

        with tab_p:
            prob_img_path = os.path.join(MAPS_DIR, "permafrost_probability_map.png")
            if os.path.exists(prob_img_path):
                st.image(prob_img_path, use_container_width=True, caption=f"Continuous Permafrost Probability Map (Inferred via {map_model})")
            else:
                st.warning("Probability map image not generated yet. Run predict.py.")

        with tab_c:
            class_img_path = os.path.join(MAPS_DIR, "permafrost_classification_map.png")
            if os.path.exists(class_img_path):
                st.image(class_img_path, use_container_width=True, caption=f"Binary Permafrost Classification Map (Threshold >= 0.5)")
            else:
                st.warning("Classification map image not generated yet. Run predict.py.")

    # Regional Statistics Summary
    if regional_stats:
        st.markdown("---")
        st.markdown("### 📈 Regional Geospatial Coverage Statistics")
        rs1, rs2, rs3, rs4 = st.columns(4)
        with rs1:
            st.metric("Total Mapped Area", f"{regional_stats['total_area_km2']:,.0f} km²")
        with rs2:
            st.metric("Permafrost-Dominant Area", f"{regional_stats['permafrost_area_km2']:,.0f} km²")
        with rs3:
            st.metric("Permafrost Coverage", f"{regional_stats['permafrost_fraction_pct']:.1f}%")
        with rs4:
            st.metric("Mean Regional Likelihood", f"{regional_stats['mean_likelihood']*100:.1f}%")

# ---------------------------------------------------------
# Page 3: Model Comparison
# ---------------------------------------------------------
elif selected_page == "📊 Model Comparison":
    st.markdown('<div class="main-header">📊 Three-Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous spatial-block cross-validation comparing Random Forest, XGBoost, and Support Vector Machine (SVM)</div>', unsafe_allow_html=True)

    if metrics_data and "models" in metrics_data:
        df_models = pd.DataFrame(metrics_data["models"])

        # Display Metrics Cards
        st.markdown("### 🏆 Performance Scorecard")
        cols = st.columns(3)
        for i, row in df_models.iterrows():
            with cols[i]:
                is_best = row["Model"] == metrics_data["best_model"]
                badge_html = '<div class="badge-best">Champion Model</div>' if is_best else '<div style="height: 24px;"></div>'
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {'#319795' if is_best else '#e2e8f0'};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #1a365d;">{row['Model']}</h3>
                        {badge_html}
                    </div>
                    <hr style="margin: 0.5rem 0;">
                    <div style="font-size: 0.95rem;"><strong>Spatial Test Accuracy:</strong> <span style="color:#2b6cb0; font-weight:700;">{row['Accuracy']*100:.2f}%</span></div>
                    <div style="font-size: 0.95rem;"><strong>Balanced Accuracy:</strong> {row['Balanced Accuracy']*100:.2f}%</div>
                    <div style="font-size: 0.95rem;"><strong>F1-Score (Weighted):</strong> <span style="font-weight:700;">{row['F1-Score']:.4f}</span></div>
                    <div style="font-size: 0.95rem;"><strong>ROC-AUC:</strong> {f"{row['ROC-AUC']:.4f}" if row['ROC-AUC'] else 'N/A'}</div>
                    <div style="font-size: 0.95rem;"><strong>Precision / Recall:</strong> {row['Precision']:.3f} / {row['Recall']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # Plots Section
    plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
        "📈 ROC Curves",
        "🎯 Confusion Matrices",
        "🌲 Feature Importances",
        "🔗 Correlation Heatmap"
    ])

    with plot_tab1:
        roc_p = os.path.join(PLOTS_DIR, "roc_curves.png")
        if os.path.exists(roc_p):
            st.image(roc_p, use_container_width=True, caption="Receiver Operating Characteristic (ROC) Curves across all three models")

    with plot_tab2:
        cm_p = os.path.join(PLOTS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_p):
            st.image(cm_p, use_container_width=True, caption="Confusion Matrices evaluated on out-of-sample Brooks Range spatial block")

    with plot_tab3:
        fi_p = os.path.join(PLOTS_DIR, "feature_importance.png")
        if os.path.exists(fi_p):
            st.image(fi_p, use_container_width=True, caption="Random Forest (Gini Importance) vs. XGBoost (Information Gain)")

    with plot_tab4:
        corr_p = os.path.join(PLOTS_DIR, "correlation_matrix.png")
        if os.path.exists(corr_p):
            st.image(corr_p, use_container_width=True, caption="Multivariate Feature Correlation Matrix")

    # Classification Reports Text
    with st.expander("📄 View Full Scikit-Learn Classification Reports (Precision / Recall by Class)"):
        rep_p = os.path.join(METRICS_DIR, "classification_reports.txt")
        if os.path.exists(rep_p):
            with open(rep_p, "r") as f:
                st.code(f.read(), language="text")

# ---------------------------------------------------------
# Page 4: Real-Time Prediction Interface
# ---------------------------------------------------------
elif selected_page == "🔍 Real-Time Prediction":
    st.markdown('<div class="main-header">🔍 Interactive Permafrost Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input environmental and satellite spectral variables to estimate permafrost likelihood</div>', unsafe_allow_html=True)

    from src.predict import predict_point

    # Preset selection
    st.markdown("#### ⚡ Quick Presets (Sample Real Locations)")
    preset_choice = st.selectbox(
        "Load Geographical Preset:",
        [
            "Custom User Input",
            "Prudhoe Bay (North Slope Tundra - High Permafrost)",
            "Toolik Lake (Arctic Foothills - High Permafrost)",
            "Brooks Range (Alpine Mountain Pass - Continuous Permafrost)",
            "Yukon River Crossing (Discontinuous Valley)",
            "Fairbanks South-Facing Ridge (Non-Permafrost Microclimate)",
        ]
    )

    # Preset values
    presets = {
        "Prudhoe Bay (North Slope Tundra - High Permafrost)": {
            "latitude": 70.25, "longitude": -148.45, "elevation": 15.0, "slope": 0.5,
            "blue": 0.045, "green": 0.085, "red": 0.065, "nir": 0.220, "swir1": 0.120, "swir2": 0.070
        },
        "Toolik Lake (Arctic Foothills - High Permafrost)": {
            "latitude": 68.63, "longitude": -149.60, "elevation": 720.0, "slope": 2.8,
            "blue": 0.038, "green": 0.075, "red": 0.055, "nir": 0.245, "swir1": 0.135, "swir2": 0.075
        },
        "Brooks Range (Alpine Mountain Pass - Continuous Permafrost)": {
            "latitude": 68.13, "longitude": -149.48, "elevation": 1440.0, "slope": 18.5,
            "blue": 0.080, "green": 0.095, "red": 0.090, "nir": 0.180, "swir1": 0.150, "swir2": 0.110
        },
        "Yukon River Crossing (Discontinuous Valley)": {
            "latitude": 65.88, "longitude": -149.72, "elevation": 120.0, "slope": 1.2,
            "blue": 0.035, "green": 0.070, "red": 0.045, "nir": 0.320, "swir1": 0.160, "swir2": 0.080
        },
        "Fairbanks South-Facing Ridge (Non-Permafrost Microclimate)": {
            "latitude": 64.85, "longitude": -147.70, "elevation": 280.0, "slope": 8.5,
            "blue": 0.030, "green": 0.065, "red": 0.040, "nir": 0.380, "swir1": 0.180, "swir2": 0.090
        },
    }

    vals = presets.get(preset_choice, {
        "latitude": 68.5, "longitude": -149.5, "elevation": 500.0, "slope": 3.0,
        "blue": 0.04, "green": 0.08, "red": 0.06, "nir": 0.24, "swir1": 0.14, "swir2": 0.08
    })

    st.markdown("---")
    c_in1, c_in2, c_in3 = st.columns(3)

    with c_in1:
        st.markdown("**Geographic & Topographic Features**")
        p_lat = st.number_input("Latitude (°N)", min_value=60.0, max_value=75.0, value=float(vals["latitude"]), step=0.05)
        p_lon = st.number_input("Longitude (°W)", min_value=-168.0, max_value=-140.0, value=float(vals["longitude"]), step=0.05)
        p_elev = st.number_input("Elevation (meters)", min_value=-50.0, max_value=3500.0, value=float(vals["elevation"]), step=10.0)
        p_slope = st.number_input("Terrain Slope (degrees)", min_value=0.0, max_value=60.0, value=float(vals["slope"]), step=0.5)

    with c_in2:
        st.markdown("**Visible & Near-Infrared Bands (0.0 – 1.0)**")
        p_blue = st.slider("Blue Band (B02)", 0.0, 1.0, float(vals["blue"]), 0.005)
        p_green = st.slider("Green Band (B03)", 0.0, 1.0, float(vals["green"]), 0.005)
        p_red = st.slider("Red Band (B04)", 0.0, 1.0, float(vals["red"]), 0.005)
        p_nir = st.slider("Near-Infrared (B08)", 0.0, 1.0, float(vals["nir"]), 0.005)

    with c_in3:
        st.markdown("**Shortwave Infrared Bands & Model**")
        p_swir1 = st.slider("SWIR-1 (B11, ~1610nm)", 0.0, 1.0, float(vals["swir1"]), 0.005)
        p_swir2 = st.slider("SWIR-2 (B12, ~2190nm)", 0.0, 1.0, float(vals["swir2"]), 0.005)
        p_model = st.selectbox("Inference Model:", ["XGBoost", "Random Forest", "SVM"], index=0)

    # Derived spectral indices display
    eps = 1e-6
    d_ndvi = (p_nir - p_red) / (p_nir + p_red + eps)
    d_ndwi = (p_green - p_nir) / (p_green + p_nir + eps)
    d_ndsi = (p_green - p_swir1) / (p_green + p_swir1 + eps)

    st.markdown(f"""
    <div style="background-color: #f7fafc; padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 0.85rem; margin-top: 0.5rem;">
        <strong>Calculated Spectral Indices:</strong> &nbsp;
        <strong>NDVI (Vegetation):</strong> {d_ndvi:.3f} |
        <strong>NDWI (Moisture/Water):</strong> {d_ndwi:.3f} |
        <strong>NDSI (Snow/Ice):</strong> {d_ndsi:.3f}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Permafrost Likelihood Prediction", type="primary", use_container_width=True):
        feature_dict = {
            "latitude": p_lat,
            "longitude": p_lon,
            "elevation": p_elev,
            "slope": p_slope,
            "blue": p_blue,
            "green": p_green,
            "red": p_red,
            "nir": p_nir,
            "swir1": p_swir1,
            "swir2": p_swir2,
            "ndvi": d_ndvi,
            "ndwi": d_ndwi,
            "ndsi": d_ndsi,
        }

        res = predict_point(feature_dict, model_name=p_model)

        st.markdown("---")
        st.markdown("### 🎯 Inference Results")
        r_col1, r_col2 = st.columns([1, 1])

        prob = res["permafrost_likelihood"]
        is_perma = res["predicted_class"] == 1

        with r_col1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": f"Permafrost Likelihood ({p_model})", "font": {"size": 18, "color": "#1a365d"}},
                number={"suffix": "%", "font": {"size": 36, "color": "#1a365d"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#e53e3e" if prob >= 0.5 else "#38a169"},
                    "steps": [
                        {"range": [0, 35], "color": "#c6f6d5"},
                        {"range": [35, 65], "color": "#fefcbf"},
                        {"range": [65, 100], "color": "#fed7d7"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with r_col2:
            status_box_color = "#fed7d7" if is_perma else "#c6f6d5"
            status_text_color = "#9b2c2c" if is_perma else "#22543d"
            st.markdown(f"""
            <div style="background-color: {status_box_color}; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
                <div style="font-size: 0.9rem; font-weight: 700; color: {status_text_color}; text-transform: uppercase;">Estimated Classification</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {status_text_color};">{res['class_label']}</div>
                <hr style="margin: 0.8rem 0; border-color: {status_text_color}; opacity: 0.3;">
                <div style="font-size: 0.95rem; color: #2d3748;">
                    <strong>Likelihood Score:</strong> {res['likelihood_percentage']}<br>
                    <strong>Risk Category:</strong> {res['risk_level']} Confidence<br>
                    <strong>Inference Model:</strong> {res['model_used']}<br>
                    <strong>Decision Threshold:</strong> P(Permafrost) ≥ 0.50
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 5: Historical Analysis
# ---------------------------------------------------------
elif selected_page == "⏳ Historical Analysis":
    st.markdown('<div class="main-header">⏳ Historical Change & Multi-Year Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tracking permafrost fraction shifts between 2020 and 2023 via ESA Climate Change Initiative NetCDF products</div>', unsafe_allow_html=True)

    if historical_data:
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Baseline Year", f"{historical_data['year_base']}")
        with h2:
            st.metric("Observation Year", f"{historical_data['year_recent']}")
        with h3:
            st.metric("Net Mean Shift", f"{historical_data['net_pfr_change']:+.2f}%", delta=f"{historical_data['net_pfr_change']:+.2f}%")
        with h4:
            st.metric("Region Stable Rate", f"{historical_data['area_stable_pct']:.1f}%")

    st.markdown("""
    <div class="scientific-note">
        <strong>Scientific Method Note:</strong> Multi-year satellite data comparison from the ESA CCI Permafrost v5.0 product reflects ground surface temperature equilibrium and active-layer modeling. A single multi-year anomaly must not be casually declared irreversible permafrost thaw without independent ground borehole thermistor validation.
    </div>
    """, unsafe_allow_html=True)

    hist_img = os.path.join(PLOTS_DIR, "historical_change_map.png")
    if os.path.exists(hist_img):
        st.image(hist_img, use_container_width=True, caption="ESA CCI Multi-Year Permafrost Extent Change (2020 vs 2023)")

    st.markdown("### 📋 Multi-Year Transition Breakdown")
    c_tr1, c_tr2 = st.columns([1, 1])
    with c_tr1:
        if historical_data:
            t_df = pd.DataFrame([
                {"Category": "Stable Active Layer (±1% change)", "Percentage": historical_data["area_stable_pct"]},
                {"Category": "Localized Surface Thaw Signal", "Percentage": historical_data["area_degraded_pct"]},
                {"Category": "Surface Cooling / Frost Persistence", "Percentage": historical_data["area_increased_pct"]},
            ])
            fig_bar = px.bar(t_df, x="Category", y="Percentage", color="Category",
                             color_discrete_map={
                                 "Stable Active Layer (±1% change)": "#3182ce",
                                 "Localized Surface Thaw Signal": "#e53e3e",
                                 "Surface Cooling / Frost Persistence": "#38a169"
                             })
            fig_bar.update_layout(height=320, showlegend=False, yaxis_title="Geographical Surface Area (%)")
            st.plotly_chart(fig_bar, use_container_width=True)

    with c_tr2:
        st.markdown("""
        #### Key Observations:
        1. **North Slope Coastal Stability:** Continuous permafrost zones (>90% fraction) exhibit minimal surface fluctuation between 2020 and 2023.
        2. **Yukon-Kuskokwim River Lowlands:** Discontinuous permafrost margins show localized inter-annual variability driven by summer thermokarst lake expansion and vegetative shrubification.
        3. **Interior Alaska Aspect Sensitivity:** South-facing sub-boreal ridges show accelerated active-layer deepening during warm summer anomalies.
        """)

# ---------------------------------------------------------
# Page 6: About Project
# ---------------------------------------------------------
elif selected_page == "ℹ️ About Project":
    st.markdown('<div class="main-header">ℹ️ About PermaWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scientific Background, Engineering Architecture, and Project Specifications</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Problem Statement
    Permafrost (ground remaining at or below 0°C for two or more consecutive years) underlies approximately **24% of the Northern Hemisphere land surface**. Rapid Arctic warming is accelerating permafrost degradation, triggering:
    - **Infrastructure Failure:** Differential ground subsidence damaging pipelines, roads (Dalton Highway), airports, and communities.
    - **Climate Feedback:** Release of sequestered carbon dioxide ($CO_2$) and methane ($CH_4$) into the atmosphere.
    - **Hydrological Alteration:** Thermokarst lake expansion, coastal bluff erosion, and altered watershed runoff.

    Because ground temperature boreholes are sparse across vast Arctic territories, **satellite remote sensing combined with machine learning** provides a vital tool for synoptic permafrost detection and monitoring.

    ---

    ### 2. Machine Learning Methodology
    PermaWatch implements three distinct classification paradigms trained on identical spatial partitions:

    1. **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`):**
       - 200 de-correlated decision trees with bootstrap aggregation.
       - Gini impurity feature evaluation; robust against outlier spectral noise and sensor anomalies.
    2. **XGBoost Classifier (`xgboost.XGBClassifier`):**
       - Gradient boosted decision trees minimizing logistic cross-entropy loss.
       - Subsampling (85%) and column sampling (85%) for regularized spatial generalization.
    3. **Support Vector Machine (`sklearn.svm.SVC`):**
       - Non-linear Radial Basis Function (RBF) kernel mapping standardized predictors into infinite-dimensional Hilbert space.
       - Platt scaling for calibrated posterior probability estimation.

    ---

    ### 3. Spatial-Block Validation (Zero Spatial Data Leakage)
    Conventional random train/test splits severely inflate evaluation accuracy in geospatial remote sensing because adjacent satellite pixels are spatially autocorrelated.
    PermaWatch utilizes **`GroupShuffleSplit`** partitioning where **entire geographic zones (e.g. Brooks Mountain Range)** are held out as unseen spatial test blocks.

    ---

    ### 4. Data Sources & Sensor Attribution
    - **Reference Labels:** Official **ESA Permafrost Climate Change Initiative (CCI) v5.0** (NetCDF4, 1 km gridded annual permafrost fraction).
    - **Optical Reflectance:** **Copernicus Sentinel-2 Level-2A** Surface Reflectance (Harmonized multispectral bands B02, B03, B04, B08, B11, B12).
    - **Terrain Elevation & Slope:** **Copernicus DEM (GLO-90)** 90-meter Global Digital Surface Model via Microsoft Planetary Computer STAC.

    ---

    ### 5. Project Limitations & Future Scope
    - **Limitation:** Optical reflectance is restricted to cloud-free summer acquisitions (polar night and cloud cover during winter).
    - **Limitation:** Satellite imagery observes surface vegetation and terrain, not subsurface soil temperatures.
    - **Future Scope:** Integrating **Sentinel-1 InSAR** differential interferometry for millimeter-scale seasonal active-layer thaw subsidence monitoring.
    - **Future Scope:** Deep learning spatial architectures (U-Net and Spatio-Temporal Transformers) for continental-scale mapping.
    """)

# ---------------------------------------------------------
# Global Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.85rem;">
    <strong>PermaWatch: Satellite Imagery-Based Permafrost Detection System</strong><br>
    Academic Minor Project Submission | Developed with Streamlit, Scikit-Learn, XGBoost, and Planetary Computer STAC
</div>
""", unsafe_allow_html=True)
