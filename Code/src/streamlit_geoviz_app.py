import streamlit as st
from streamlit_geoviz_state_manager import initialize_state
from node_filter import NodeFilter

# Initialize session state variables
initialize_state()
bload = st.session_state.bload
plotter = st.session_state.plotter
geonodes, geodata = st.session_state.geo_data

# --- Streamlit UI --- #
st.title(f"Graph Visualization {bload.foreground_name}")

##  User selections ##
show_edges = st.checkbox("Show Flows", value=True)
show_geolocate = st.checkbox("Show Only Geolocated", value=False)

## Filtering ## 
activity_types = list(geonodes['activity_type'].unique())  # Unique activity types
selected_activities = [activity for activity in activity_types if st.checkbox(f"Show '{activity}' nodes", value=True)]

min_betweenness = st.slider("Min Betweenness Centrality", 0.0, 1.0, 0.0, step=0.05)
min_norm_betweenness = st.slider("Min Percentile in Betweenness Centrality Distribtuion", 0.0, 1.0, 0.0, step=0.05)

filter_config = {
    "filter_by_betweenness" : min_betweenness,
    "filter_by_activity_type" : selected_activities,
    "filter_by_normalized_betweenness" : min_norm_betweenness,
    "filter_by_geolocation" : show_geolocate,
}

fig = plotter.create_figure(filter_config=filter_config, show_edges=show_edges)
st.plotly_chart(fig, use_container_width=True)