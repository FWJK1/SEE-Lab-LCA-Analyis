import numpy as np
import pandas as pd
from shapely.geometry import Point

class OffshorePointGenerator:
    def __init__(self, country_shapefile_frame, geo_nodes, subedges):
        self.country_shapefile_frame = country_shapefile_frame
        self.geo_nodes = geo_nodes
        self.subedges = subedges

    def non_geo_point(self, row):
        """Finds an offshore point near a connected geolocated node or generates a random offshore point."""
        connected_nodes = self.get_connected_geolocated_nodes(row["node"])

        if not connected_nodes:
            return self.get_random_offshore_point()

        nearest_geo_node = connected_nodes[0]
        base_lon, base_lat = nearest_geo_node['Longitude'], nearest_geo_node['Latitude']

        for _ in range(100):  # Try up to 100 times
            shift_lon, shift_lat = self.generate_shift()
            new_lon = base_lon + shift_lon
            new_lat = base_lat + shift_lat

            if not self.is_on_land(new_lon, new_lat):
                return Point(new_lon, new_lat)

        return self.get_random_offshore_point()

    def get_random_offshore_point(self):
        """Generates a completely random offshore point."""
        lon, lat = np.random.uniform(-180, 180), np.random.uniform(-85, 85)
        while self.is_on_land(lon, lat):
            lon, lat = np.random.uniform(-180, 180), np.random.uniform(-85, 85)
        return Point(lon, lat)

    def generate_shift(self):
        """Generates a small random offshore shift."""
        shift_lon = np.random.uniform(0.5, 2.0) * (-1 if np.random.rand() > 0.5 else 1)
        shift_lat = np.random.uniform(0.5, 2.0) * (-1 if np.random.rand() > 0.5 else 1)
        return shift_lon, shift_lat

    def is_on_land(self, lon, lat):
        """Checks if a point (lon, lat) is on land."""
        point = Point(lon, lat)
        return self.country_shapefile_frame.contains(point).any()

    def get_connected_geolocated_nodes(self, node):
        """
        Returns a list of geolocated nodes that are directly connected to the given non-geolocated node.
        Assumes `self.G` is a NetworkX graph where node attributes store location data.
        """
        connected_edges = self.subedges[
        (self.subedges['source_node'] == node) | (self.subedges['target_node'] == node)
        ]
        connected_nodes = list(set(connected_edges['source_node']).union(connected_edges['target_node']))
        connected_nodes = self.geo_nodes[self.geo_nodes['node'].isin(connected_nodes)]
        connected_nodes = connected_nodes[connected_nodes['geometry'].notna()]

        return [{"Node": n, "Longitude": node_info["random_point"].x, "Latitude": node_info["random_point"].y}
                    for n, node_info in connected_nodes.iterrows()]    

