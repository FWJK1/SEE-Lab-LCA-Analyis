import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors


"""
idea for non-geo visualization pipeline

Store the graph and the dataframes separately (each still cached)

Apply same  filtering logic on node frame
Create a subgraph using something like G.subgraph(filtered_nodes)
Plot the subgraph  using Plotly or another visualization tool.
"""

class GeoPlotter:
    def __init__(self, geol, nodes):
        """
        Initializes the GeoPlotter class with geographic and node-edge data.

        Args:
            geol (object): An object containing the geographic data (e.g., shapefile) for plotting.
        """
        self.geo_traces = self.compute_geo_traces(geol)
        activity_types  = nodes['activity_type'].unique().tolist()
        palette = sns.color_palette("Set2", len(activity_types))
        self.activity_color_map = {
            activity: mcolors.to_hex(palette[i])  # Convert each Seaborn RGB to hex
            for i, activity in enumerate(activity_types)
        }


    def compute_geo_traces(self, geol):
        """
        Precomputes the static geographic boundaries from a shapefile and returns them as Plotly traces.

        Args:
            geol (object): An GeoLocator object containing the geographic data (e.g., shapefile).

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
    
    def compute_node_traces(self, active_nodes):
        """
        Computes the node trace for visualization from the filtered dataframe

        Args:
            filter_config (dict): A dictionary of filters to be applied to the node DataFrame.

        Returns:
            list: A list of Plotly scatter traces for the active nodes, each trace representing a different activity type.
            pandas.DataFrame: A filtered DataFrame containing only the active nodes.
        """
        traces = []
        activity_types = active_nodes['activity_type'].unique()
        for activity in activity_types:
            filtered_df = active_nodes[active_nodes['activity_type'] == activity]
            node_trace = go.Scatter(
                x=filtered_df.geometry.x,
                y=filtered_df.geometry.y,
                mode="markers",
                name=str(activity), 
                marker=dict(
                    size=10, 
                    color=self.activity_color_map[activity],
                    opacity=0.9),
                text=filtered_df["location"] + " : " + filtered_df["name"],
                hoverinfo="text",
            )
            traces.append(node_trace)
        return traces

    
    def compute_edge_trace(self, filtered_edges):
        """
        Computes the edge trace for visualization based on the active nodes.

        Args:
            active_nodes (pandas.DataFrame): A DataFrame containing the nodes that passed the filters.

        Returns:
            go.Scatter: A Plotly scatter trace for the edges, with lines connecting active nodes.
        """
        if filtered_edges.empty:
            return go.Scatter(x=[], y=[], mode='lines', line=dict(color='grey', width=0.3), showlegend=False)
    
        all_coords = filtered_edges['geometry'].map(lambda line: list(line.coords))  # Extract all edge coordinates
        flat_x = np.concatenate([np.append([point[0] for point in coords], None) for coords in all_coords])
        flat_y = np.concatenate([np.append([point[1] for point in coords], None) for coords in all_coords])
        edge_trace = go.Scatter(
            x=flat_x,
            y=flat_y,
            mode='lines',
            line=dict(color='grey', width=0.4),
            opacity=0.8,
            hoverinfo='none',
            showlegend=False
        )
        return edge_trace

    def create_figure(self, filtered_nodes, filtered_edges, show_edges=True, **kwargs):
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
        for trace in self.compute_node_traces(filtered_nodes):
            fig.add_trace(trace)
        if show_edges:
            fig.add_trace(self.compute_edge_trace(filtered_edges))
        return fig