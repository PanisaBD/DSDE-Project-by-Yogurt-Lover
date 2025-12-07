import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Bangkok Traffy Analysis", layout="wide")

st.title("Bangkok Traffy Fondue Reports Analysis")

# Optimized data loading - only essential columns
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "bangkok_traffy_cleaned.csv",
            usecols=['ticket_id', 'type', 'coords', 'district', 'subdistrict', 'organization',
                     'state', 'timestamp', 'last_activity']
        )
    except:
        st.error("❌ Could not load bangkok_traffy_cleaned.csv")
        return None

    # Parse coordinates
    coords_split = df['coords'].str.split(',', expand=True)
    df['lon'] = pd.to_numeric(coords_split[0], errors='coerce')
    df['lat'] = pd.to_numeric(coords_split[1], errors='coerce')
    
    # Timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['last_activity'] = pd.to_datetime(df['last_activity'], errors='coerce')

    # Solve time
    df['solve_time_days'] = (df['last_activity'] - df['timestamp']).dt.total_seconds() / (24 * 3600)

    # Clean type
    df['type_clean'] = df['type'].str.replace('[{}]', '', regex=True)

    # Drop invalid rows
    df = df.dropna(subset=['lon', 'lat'])

    return df


# Get aggregated statistics
@st.cache_data
def get_stats(df):
    return {
        'district_counts': df['district'].value_counts(),
        'subdistrict_counts': df['subdistrict'].value_counts(),
        'type_counts': df['type_clean'].value_counts(),
        'org_counts': df['organization'].value_counts(),
        'total': len(df)
    }

# Sidebar
with st.spinner("Loading data..."):
    df = load_data()

if df is None or len(df) == 0:
    st.stop()

st.success(f"✅ Loaded {len(df):,} tickets")
# Get pre-computed stats
stats = get_stats(df)

# Filters
st.sidebar.header("🔍 Filters")

# District filter - REQUIRED
districts = stats['district_counts'].index.tolist()
sel_dist = st.sidebar.multiselect(
    "Select District(s) *Required*", 
    districts,
    max_selections=3,
    help="Select 1-3 districts (required)"
)

if not sel_dist:
    st.info("👈 Select 1-3 districts from the sidebar to start analysis")
    st.stop()

# Filter data by district first
filt = df[df['district'].isin(sel_dist)].copy()

# Other filters based on filtered data
subdists = filt['subdistrict'].unique().tolist()
sel_sub = st.sidebar.multiselect("Subdistrict", sorted(subdists))

types = filt['type_clean'].unique().tolist()
sel_type = st.sidebar.multiselect("Type", sorted(types)[:20])  # Limit options

# Apply remaining filters
if sel_sub:
    filt = filt[filt['subdistrict'].isin(sel_sub)]
if sel_type:
    filt = filt[filt['type_clean'].isin(sel_type)]

# Date filter - simplified
filt['month'] = filt['timestamp'].dt.to_period('M')
months = sorted(filt['month'].unique().astype(str))
if len(months) > 0:
    sel_months = st.sidebar.multiselect("Select Month(s)", months, default=months[-3:] if len(months) >= 3 else months)
    if sel_months:
        filt = filt[filt['month'].astype(str).isin(sel_months)]

# Limit map data
MAP_LIMIT = 5000
map_data = filt.head(MAP_LIMIT)

# Map type selection
st.sidebar.header("Map Options")
map_type = st.sidebar.radio("Map Type", ["Heatmap", "Cluster"], index=0)

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tickets", f"{len(filt):,}")
col2.metric("Districts", filt['district'].nunique())
col3.metric("Subdistricts", filt['subdistrict'].nunique())
col4.metric("Types", filt['type_clean'].nunique())

# Calculate average solve time
avg_solve_time = filt['solve_time_days'].mean()
col5.metric("Avg Solve Time", f"{avg_solve_time:.1f} days")

# Map
st.header(f"📍 {map_type} Map (Report on Traffy Fondoe)")
if len(filt) > MAP_LIMIT:
    st.warning(f"⚠️ Showing first {MAP_LIMIT:,} tickets on map (total: {len(filt):,})")

center_lat = map_data['lat'].mean()
center_lon = map_data['lon'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

if map_type == "Heatmap":
    heat_data = map_data[['lat', 'lon']].values.tolist()
    HeatMap(heat_data, radius=15, blur=25, min_opacity=0.5, max_zoom=18).add_to(m)
else:  # Cluster
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in map_data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(f"""
                <b>District:</b> {row['district']}<br>
                <b>Subdistrict:</b> {row['subdistrict']}<br>
                <b>Type:</b> {row['type_clean']}<br>
                <b>Organization:</b> {row['organization']}
            """, max_width=250),
            tooltip=f"{row['district']} - {row['subdistrict']}"
        ).add_to(marker_cluster)

st_folium(m, width=1400, height=500)

# Analytics
st.header("Analytics")
col1, col2, col3 = st.columns(3)

# --- Districts ---
with col1:
    st.subheader("Districts")
    dist_cnt = filt['district'].value_counts().head(10).reset_index()
    dist_cnt.columns = ['District', 'Count']
    fig1 = px.bar(dist_cnt, x='District', y='Count', color='Count',
                  color_continuous_scale='Reds')
    fig1.update_layout(
        xaxis_tickangle=-45,
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- Top Types ---
with col2:
    st.subheader("Top Types")
    type_cnt = filt['type_clean'].value_counts().head(10).reset_index()
    type_cnt.columns = ['Type', 'Count']
    fig2 = px.bar(type_cnt, x='Type', y='Count', color='Count',
                  color_continuous_scale='Blues')
    fig2.update_layout(
        xaxis_tickangle=-45,
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Subdistricts ---
with col3:
    st.subheader("Subdistricts")
    sub_cnt = filt['subdistrict'].value_counts().head(10).reset_index()
    sub_cnt.columns = ['Subdistrict', 'Count']
    fig3 = px.bar(sub_cnt, y='Subdistrict', x='Count', orientation='h',
                  color='Count', color_continuous_scale='Greens')
    fig3.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)
# Timeline
st.subheader("Timeline")
timeline = filt.groupby(filt['month'].astype(str)).size().reset_index(name='count')
timeline.columns = ['Month', 'Count']
fig5 = px.line(timeline, x='Month', y='Count', markers=True)
fig5.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig5, use_container_width=True)

# Solve Time Analysis
st.header("Solve Time Analysis")

col_st1, col_st2 = st.columns(2)

with col_st1:
    solve_by_dist = filt.groupby('district')['solve_time_days'].mean().sort_values(ascending=False).head(10).reset_index()
    solve_by_dist.columns = ['District', 'Avg Days']
    fig_solve1 = px.bar(solve_by_dist, x='District', y='Avg Days', 
                        color='Avg Days', color_continuous_scale='RdYlGn_r',
                        title='Average Resolution Time by District')
    fig_solve1.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_solve1, use_container_width=True)

with col_st2:
    solve_by_type = filt.groupby('type_clean')['solve_time_days'].mean().sort_values(ascending=False).head(10).reset_index()
    solve_by_type.columns = ['Type', 'Avg Days']
    fig_solve2 = px.bar(solve_by_type, y='Type', x='Avg Days', orientation='h',
                        color='Avg Days', color_continuous_scale='RdYlGn_r',
                        title='Average Resolution Time by Type')
    fig_solve2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_solve2, use_container_width=True)

# Solve time distribution
st.subheader("Distribution of Solve Times")
fig_solve3 = px.histogram(filt, x='solve_time_days', nbins=50,
                          title='Distribution of Resolution Times',
                          labels={'solve_time_days': 'Days to Resolve', 'count': 'Number of Tickets'})
fig_solve3.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_solve3, use_container_width=True)
# ============================================================
# 🔍 District–Population Density / Housing Density Comparison
# ============================================================

st.header("District Density & Solve Time Comparison")

@st.cache_data
def load_density():
    try:
        return pd.read_csv("clean_combined_data.csv")
    except:
        st.error(" clean_combined_data.csv not found")
        return None

density_df = load_density()
if density_df is None:
    st.stop()

# -----------------------------
# 1) Convert Thai short year → Gregorian
# Example: 65 → 2022
# -----------------------------
density_df['year'] = density_df['year'].astype(int) + 1957  # (2500 - 543 = 1957)

# -----------------------------
# 2) Month already numeric "01"–"12", use directly
# -----------------------------
density_df['month'] = density_df['month'].astype(str).str.zfill(2)

# -----------------------------
# 3) Create YYYY-MM for merging
# -----------------------------
density_df['ym'] = density_df['year'].astype(str) + "-" + density_df['month']

# -----------------------------
# 4) Prepare Traffy YYYY-MM
# -----------------------------
filt['ym_full'] = filt['timestamp'].dt.strftime('%Y-%m')

# -----------------------------
# 5) Merge Traffy + Density
# -----------------------------
merge_df = pd.merge(
    filt,
    density_df,
    left_on=['district', 'ym_full'],
    right_on=['district', 'ym'],
    how='left'
)

# -----------------------------
# 6) Aggregation / Comparison
# -----------------------------
compare_df = merge_df.groupby(['district', 'ym_full']).agg({
    'solve_time_days': 'mean',
    'ticket_id': 'count',
    'population_density': 'mean',
    'housing_density': 'mean'
}).reset_index()

compare_df.columns = [
    "District",
    "Month",
    "Avg_Solve_Time",
    "Ticket_Count",
    "Population_Density",
    "Housing_Density"
]

st.subheader("District Comparison Table")
st.dataframe(compare_df, use_container_width=True)

# ============================================================
# Scatter Plots
# ============================================================
st.subheader("Relationship: Solve Time vs Density")

fig_corr = px.scatter(
    compare_df,
    x="Population_Density",
    y="Avg_Solve_Time",
    color="District",
    size="Ticket_Count",
    hover_data=["Month"],
    title="Population Density vs Avg Solve Time"
)
st.plotly_chart(fig_corr, use_container_width=True)

fig_corr2 = px.scatter(
    compare_df,
    x="Housing_Density",
    y="Avg_Solve_Time",
    color="District",
    size="Ticket_Count",
    hover_data=["Month"],
    title="Housing Density vs Avg Solve Time"
)
st.plotly_chart(fig_corr2, use_container_width=True)



# Summary stats table
st.header("Summary Statistics")
summary_df = pd.DataFrame({
    'Metric': ['Total Tickets', 'Districts', 'Subdistricts', 'Organizations', 'Ticket Types'],
    'Count': [
        len(filt),
        filt['district'].nunique(),
        filt['subdistrict'].nunique(),
        filt['organization'].nunique(),
        filt['type_clean'].nunique()
    ]
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.caption(f"Total: {stats['total']:,} | Filtered: {len(filt):,} | Showing: {len(map_data):,} on map")