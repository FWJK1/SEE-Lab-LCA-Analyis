from geo_plot  import GeoPlotter
from interactive_graph import log_time
from brightway_loader import BrightwayLoader
from non_geo_viz import NonGeoVisualizer
from LCA_Analyzer import lcaAnalyzer
from subgraph_gen import SubgraphProcessor
from geo_locator import GeoLocator
from shared_funcs import get_random_node

## third party packages
import plotly.graph_objects as go
import streamlit as st

@st.cache_resource
def get_graph():
    return BrightwayLoader()

@st.cache_resource
def graph_lca(bload):
    # run lca
    lca_analyzer = lcaAnalyzer()
    lca_analyzer.run_lca()
    nodes, edges = lca_analyzer.get_nodes_edges_list_from_lca()

    ## Build Subgraph and create frames 
    sp= SubgraphProcessor(bload)
    subgraph = sp.build_subgraph(nodes, edges=edges, levels=0)
    subnodes, subedges = sp.frames_from_subgraph(subgraph)

    ## geolocate data
    geol = GeoLocator(bload)
    return geol.geolocate(subnodes, subedges)

def graph_random_node(bload):
    # ## Build Subgraph and create frames       
    sp = SubgraphProcessor(bload)
    subgraph = sp.build_subgraph([get_random_node(bload.G)])
    subnodes, subedges = sp.frames_from_subgraph(subgraph)

    ## geolocate data
    geol = GeoLocator(bload)
    subnodes, subedges = geol.geolocate(subnodes, subedges)

@st.cache_resource
def precompute_traces(_ig, _nodes_df, _edges_df):
    """Precompute traces for faster rendering in Streamlit."""
    geo_traces, node_traces, edge_trace = [], {}, None

    # Geographic boundaries
    for geometry in _ig.country_reference_gdf.geometry:
        if geometry.geom_type == 'Polygon':
            x, y = geometry.exterior.xy
            geo_traces.append(go.Scatter(x=x, y=y, mode='lines', line=dict(color='lightgrey', width=2), showlegend=False))
        elif geometry.geom_type == 'MultiPolygon':
            for poly in geometry.geoms:
                x, y = poly.exterior.xy
                geo_traces.append(go.Scatter(x=x, y=y, mode='lines', line=dict(color='lightgrey', width=2), showlegend=False))

    # Node traces
    activity_types = _nodes_df['activity_type'].unique()
    for activity in activity_types:
        filtered_df = _nodes_df[_nodes_df['activity_type'] == activity]
        node_traces[activity] = go.Scatter(
            x=filtered_df.geometry.x,
            y=filtered_df.geometry.y,
            mode='markers',
            marker=dict(size=10, opacity=0.9),
            name=activity,
            text=filtered_df['name'],
            hoverinfo='text',
        )

    # Edge traces
    active_node_ids = set(_nodes_df['node'])
    edge_x, edge_y = [], []
    for _, edge in _edges_df.iterrows():
        if edge['source_node'] in active_node_ids and edge['target_node'] in active_node_ids:
            line = edge.geometry
            edge_x.extend([point[0] for point in line.coords] + [None])
            edge_y.extend([point[1] for point in line.coords] + [None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='lightblue', width=0.3), opacity=0.7, name='Flows')

    return GeoPlotter(geo_traces, node_traces, edge_trace)

# --- Streamlit App ---
bload = get_graph()
st.title(f"Graph Visualization {bload.foreground_name}")
geo_nodes, geo_edges = graph_lca(bload)
plotter = precompute_traces(ig, geo_nodes, geo_edges)

# User selections
visible_activities = [activity for activity in plotter.node_traces.keys() if st.checkbox(f"Show '{activity}' nodes", value=True)]
show_edges = st.checkbox("Show Flows", value=True)

# Generate and display the figure
fig = plotter.create_figure(visible_activities, show_edges)
st.plotly_chart(fig, use_container_width=True)
