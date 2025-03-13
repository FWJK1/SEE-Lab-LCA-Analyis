import plotly.graph_objects as go
import pandas as pd
from node_filter import NodeFilter

class GeoPlotter:
    def __init__(self, geol, node_df, edge_df):
        """
        Initializes the GeoPlotter class with geographic and node-edge data.

        Args:
            geol (object): An object containing the geographic data (e.g., shapefile) for plotting.
            node_df (pandas.DataFrame): DataFrame containing nodes' information (e.g., coordinates, attributes).
            edge_df (pandas.DataFrame): DataFrame containing edges' information (e.g., source, target nodes, geometry).
        """
        self.geo_traces = self.compute_geo_traces(geol)
        self.NodeFilter = NodeFilter(node_df, edge_df)

    def compute_geo_traces(self, geol):
        """
        Precomputes the static geographic boundaries from a shapefile and returns them as Plotly traces.

        Args:
            geol (object): An object containing the geographic data (e.g., shapefile).

        Returns:
            list: A list of Plotly `Scattergl` traces representing the geographic boundaries.
        """
        geo_traces = []
        x_all, y_all = [], []
        for geometry in geol.country_shapefile_frame.geometry:
            if geometry.geom_type == 'Polygon':
                x, y = geometry.exterior.xy
                x_all.append(list(x))
                y_all.append(list(y))
            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    x, y = poly.exterior.xy
                    x_all.append(list(x))
                    y_all.append(list(y))

        for x, y in zip(x_all, y_all):
            geo_traces.append(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='lightgrey', width=2),
                showlegend=False
            ))
        return geo_traces

    def create_figure(self, filter_config={}, show_edges=True):
        """
        Builds and returns a Plotly figure, including geographic boundaries, filtered nodes, and edges.

        Args:
            filter_config (dict, optional): A dictionary of filters to apply to the node DataFrame (default is an empty dictionary).
            show_edges (bool, optional): Whether to display edges between nodes (default is True).

        Returns:
            plotly.graph_objects.Figure: A Plotly figure containing the plotted geographic boundaries, nodes, and edges.
        """
        fig = go.Figure()
        for trace in self.geo_traces:
            fig.add_trace(trace)
        node_traces, active_nodes = self.NodeFilter.compute_node_trace(filter_config)
        for trace in node_traces:
            fig.add_trace(trace)

        if show_edges:
            fig.add_trace(self.NodeFilter.compute_edge_trace(active_nodes))
        return fig
