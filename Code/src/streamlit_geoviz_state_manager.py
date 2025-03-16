import streamlit as st
from geo_plot import GeoPlotter
from brightway_loader import BrightwayLoader
from LCA_Analyzer import lcaAnalyzer
from subgraph_gen import SubgraphProcessor
from geo_locator import GeoLocator
from shared_funcs import get_random_node
from node_filter import NodeFilter
from nx_plot import NxPlotter


def initialize_state(subgraph_mode='lca', geomode="offshore", geoprojection="4327"):
    """Initialize session state variables in a fixed order, ensuring dependencies are met."""

    st.session_state.bload = BrightwayLoader()
    st.session_state.subproc = SubgraphProcessor(st.session_state.bload)
    st.session_state.geolocator = GeoLocator(st.session_state.bload, geomode=geomode, projection=geoprojection)
    
    geo_nodes, geo_edges = load_subgraph_data(subgraph_mode, st.session_state.subproc)
    st.session_state.geo_data = geo_nodes, geo_edges

    st.session_state.geoplotter = GeoPlotter(st.session_state.geolocator, geo_nodes)
    st.session_state.nx_plotter = NxPlotter(st.session_state.subproc.subgraph, geo_nodes)
    st.session_state.node_filter = NodeFilter(geo_nodes, geo_edges)

    st.session_state.plotters = {
        "GeoNetwork" : st.session_state.geoplotter,
        "GeoHeatmap": False,
        "Standard Network" : st.session_state.nx_plotter 
    }

    st.session_state.activity_types = list(geo_nodes['activity_type'].unique())
    st.session_state.max_in_degree = geo_nodes['in_degree'].max()
    st.session_state.max_out_degree = geo_nodes['out_degree'].max()

    st.session_state.title = (
        st.session_state.bload.foreground_name if subgraph_mode == "lca" else "random node"
    )


def load_subgraph_data(mode, sp):
    """Runs LCA analysis and returns geolocated nodes and edges."""
    if mode == "lca":
        lca_analyzer = lcaAnalyzer()
        lca_analyzer.run_lca()
        nodes, edges = lca_analyzer.get_nodes_edges_list_from_lca()
        subgraph = sp.build_subgraph(nodes, edges=edges, levels=0)
    else:
        subgraph = sp.build_subgraph([get_random_node(st.session_state.bload.G)])

    subnodes, subedges = sp.frames_from_subgraph(subgraph)
    return st.session_state.geolocator.geolocate(subnodes, subedges)
