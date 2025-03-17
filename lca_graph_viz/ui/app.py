## standard python packages
import argparse
import io


## 3rd party packages
import streamlit as st

## homebrew packages
from streamlit_state_manager import initialize_state

## helpful func to make streamlit ui a little less annoying
def custom_radio(label, options, value):
    return st.radio(label, options, index=options.index(value))

if 'initialized' not in st.session_state:
    # Parse command-line arguments for initial state (don't change with streamlit interactivity)
    parser = argparse.ArgumentParser(description="GeoLocator command-line configuration.")
    parser.add_argument('--subgraph_mode', type=str, default='random', choices=['random', 'lca'], 
                        help="Choose the subgraph mode (random or specific). Default is 'random'.")
                        
    parser.add_argument('--geomode', type=str, default='offshore', choices=['offshore', 'antarctica'], 
                        help="Choose the geolocation mode ('offshore' or 'antarctica'). Default is 'offshore'.")
    parser.add_argument('--geoprojection', type=str, default=4327, 
                        help="Choose the ESPG projection for the map. Default is 4327 (World Geodetic System 1984)")
    args = parser.parse_args()

    initialize_state(subgraph_mode=args.subgraph_mode, geomode=args.geomode, geoprojection=args.geoprojection)
    st.session_state.initialized = True

## vars for filtering
activity_types = st.session_state.activity_types
max_in_degree = st.session_state.max_in_degree
max_out_degree = st.session_state.max_out_degree


# --- Streamlit UI --- # 
st.title(f"Graph Visualization for {st.session_state.title}")


# Sidebar for Filters
with st.sidebar:
    st.header("Display Options")
    show_edges = st.checkbox("Show Flows", value=True)
    show_geolocate = st.checkbox("Show Only Geolocated", value=False)
    viz_version = custom_radio("Vizualization Method", options=['GeoNetwork',  'GeoHeatmap', 'Standard Network'], value="GeoNetwork")
    nx_layout = custom_radio("NetworkX Position Algorithm", value="spring", options = [
        "spring", "kamada_kawai","spectral","circular","random"
    ])
    st.header("Filtering")
    selected_activities = st.multiselect("Select activity types to display:", activity_types, default=activity_types)
    min_betweenness = st.slider("Min Betweenness Centrality", 0.0, 1.0, 0.0, step=0.05)
    min_norm_betweenness = st.slider("Min Percentile in Betweenness Centrality Distribution", 0.0, 1.0, 0.0, step=0.05)
    min_in_degree = st.slider("Min Node In-Degree", 0, max_in_degree, 0)
    min_out_degree = st.slider("Min Node Out-Degree", 0, max_out_degree, 0)


## filtering from ui
filter_config = {
    "filter_by_betweenness" : min_betweenness,
    "filter_by_activity_type" : selected_activities,
    "filter_by_normalized_betweenness" : min_norm_betweenness,
    "filter_by_geolocation" : show_geolocate,
    "filter_by_in_degree" : min_in_degree,
    "filter_by_out_degree" : min_out_degree,
}

subnodes, subedges =  st.session_state.node_filter.filter_frames(filter_config)

## actual plotting
plotter = st.session_state.plotters.get(viz_version)
fig = plotter.create_figure(filtered_nodes=subnodes, filtered_edges=subedges, show_edges=show_edges,
                            layout=nx_layout)

if viz_version == 'GeoHeatmap':
    st.pyplot(fig)
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    # print("fig saved")
    # st.image(buf, use_column_width=True)
else:
    st.plotly_chart(fig)