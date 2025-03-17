## 3rd party packages 
import pandas as pd

class NodeFilter:
    def __init__(self, node_df, edge_df):
        """
        Initializes the NodeFilter class with node and edge dataframes. This class is for filtering node and edge dataframes regardless of what they're eventually used for. 

        Args:
            node_df (pandas.DataFrame): DataFrame containing nodes' information (e.g., betweenness, activity type, coordinates).
            edge_df (pandas.DataFrame): DataFrame containing edges' information (e.g., source, target nodes, geometry).
        """
        self.node_df = node_df
        self.edge_df = edge_df


    def apply_filters(self, filter_config):
        """
        Applies multiple filters to the node DataFrame in a vectorized way.

        Args:
            filter_config (dict): A dictionary where keys are filter names and values are their corresponding parameters.

        Returns:
            pandas.DataFrame: A filtered DataFrame containing only the nodes that pass the given filters.
        """
        # Start with all True mask
        mask = pd.Series(True, index=self.node_df.index) 
        
        # mapping of filter names to vectorized functions
        vectorized_filters = {
            "filter_by_betweenness": lambda df, v: df["betweenness_centrality"] >= v,
            "filter_by_geolocation": lambda df, v: df["is_geolocated"] if v else True,
            "filter_by_activity_type": lambda df, v: df["activity_type"].isin(set(v)) if v else True,
            "filter_by_normalized_betweenness": lambda df, v: df["normalized_betweenness"] >= v,
            "filter_by_in_degree": lambda df, v: df["in_degree"] >= v,
            "filter_by_out_degree": lambda df, v: df["out_degree"] >= v,
        }
        for filter_name, value in filter_config.items():
            if filter_name in vectorized_filters:
                mask &= vectorized_filters[filter_name](self.node_df, value)

        return self.node_df[mask]

    def filter_frames(self, filter_config):
            filtered_nodes = self.apply_filters(filter_config=filter_config)
            active_nodes = filtered_nodes['node'].tolist()
            filtered_edges = self.edge_df[
            (self.edge_df['target_node'].isin(active_nodes)) & (self.edge_df['source_node'].isin(active_nodes))
            ]
            return filtered_nodes, filtered_edges