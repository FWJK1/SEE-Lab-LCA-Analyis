import streamlit as st
from geo_plot import GeoPlotter
from brightway_loader import BrightwayLoader
from LCA_Analyzer import lcaAnalyzer
from subgraph_gen import SubgraphProcessor
from geo_locator import GeoLocator

def initialize_state():
    """Initialize session state variables if not already set."""
    if "bload" not in st.session_state:
        st.session_state.bload = BrightwayLoader()
    if "geolocator" not in st.session_state:
        st.session_state.geolocator = GeoLocator(st.session_state.bload)
    if "geo_data" not in st.session_state:
        st.session_state.geo_data = load_graph_data()
    if "plotter" not in st.session_state:
        geo_nodes, geo_edges = st.session_state.geo_data
        st.session_state.plotter = GeoPlotter(st.session_state.geolocator, geo_nodes, geo_edges)

@st.cache_resource
def load_graph_data():
    """Runs LCA analysis and returns geolocated nodes and edges."""
    lca_analyzer = lcaAnalyzer()
    lca_analyzer.run_lca()
    nodes, edges = lca_analyzer.get_nodes_edges_list_from_lca()
    
    sp = SubgraphProcessor(st.session_state.bload)
    subgraph = sp.build_subgraph(nodes, edges=edges, levels=0)
    subnodes, subedges = sp.frames_from_subgraph(subgraph)
    
    return st.session_state.geolocator.geolocate(subnodes, subedges)
