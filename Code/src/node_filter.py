import pandas as pd
import plotly.graph_objects as go
import numpy as np

class NodeFilter:
    def __init__(self, node_df, edge_df):
        """
        Initializes the NodeFilter class with node and edge dataframes.

        Args:
            node_df (pandas.DataFrame): DataFrame containing nodes' information (e.g., betweenness, activity type, coordinates).
            edge_df (pandas.DataFrame): DataFrame containing edges' information (e.g., source, target nodes, geometry).
        """
        self.node_df = node_df
        self.edge_df = edge_df


    def filter_by_betweenness(self, row, min_betweenness):
        return row['betweenness_centrality'] >= min_betweenness
    
    
    def filter_by_geolocation(self, row, check):
        if check:
            return row['is_geolocated']
        else:
            return True

    def filter_by_activity_type(self, row, activity_types):
        return row['activity_type'] in activity_types if activity_types else True
    
    def filter_by_normalized_betweenness(self, row, min_norm_betweenness):
        return row['normalized_betweenness'] >= min_norm_betweenness

    def apply_filters(self, filter_config):
        """
        Applies multiple filters to the node DataFrame.

        Args:
            filter_config (dict): A dictionary where keys are filter names (methods) and values are their corresponding parameters.

        Returns:
            pandas.DataFrame: A filtered DataFrame containing only the nodes that pass the given filters.
        """
        mask = pd.Series([True] * len(self.node_df))  # Start with a mask where all nodes are included
        for filter_name, value in filter_config.items():
            filter_func = getattr(self, filter_name)
            mask &= self.node_df.apply(lambda row: filter_func(row, value), axis=1)  # Apply filter logic to the rows
        return self.node_df[mask]
    
    def compute_node_trace(self, filter_config):
        """
        Computes the node trace for visualization based on the applied filters.

        Args:
            filter_config (dict): A dictionary of filters to be applied to the node DataFrame.

        Returns:
            list: A list of Plotly scatter traces for the active nodes, each trace representing a different activity type.
            pandas.DataFrame: A filtered DataFrame containing only the active nodes.
        """
        active_nodes = self.apply_filters(filter_config)
        traces = []
        activity_types = active_nodes['activity_type'].unique()
        for activity in activity_types:
            filtered_df = active_nodes[active_nodes['activity_type'] == activity]
            node_trace = go.Scatter(
                x=filtered_df.geometry.x,
                y=filtered_df.geometry.y,
                mode="markers",
                name=str(activity), 
                marker=dict(size=10, opacity=0.9),
                text=filtered_df["location"] + " : " + filtered_df["name"],
                hoverinfo="text",
            )
            traces.append(node_trace)
        return traces, active_nodes

    
    def compute_edge_trace(self, active_nodes):
        """
        Computes the edge trace for visualization based on the active nodes.

        Args:
            active_nodes (pandas.DataFrame): A DataFrame containing the nodes that passed the filters.

        Returns:
            go.Scatter: A Plotly scatter trace for the edges, with lines connecting active nodes.
        """
        active_nodes = active_nodes['node'].tolist()
        filtered_edges = self.edge_df[
            (self.edge_df['target_node'].isin(active_nodes)) & (self.edge_df['source_node'].isin(active_nodes))
            ]
        if filtered_edges.empty:
            return go.Scatter(x=[], y=[], mode='lines', line=dict(color='lightblue', width=0.3), showlegend=False)
    
        all_coords = filtered_edges['geometry'].map(lambda line: list(line.coords))  # Extract all edge coordinates
        flat_x = np.concatenate([np.append([point[0] for point in coords], None) for coords in all_coords])
        flat_y = np.concatenate([np.append([point[1] for point in coords], None) for coords in all_coords])
        edge_trace = go.Scatter(
            x=flat_x,
            y=flat_y,
            mode='lines',
            line=dict(color='lightblue', width=0.4),
            opacity=0.8,
            hoverinfo='none',
            showlegend=False
        )
        return edge_trace
