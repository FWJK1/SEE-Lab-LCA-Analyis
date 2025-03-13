## python packages 
from pathlib import Path
import random


## 3rd party packages 
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString


## homebrew packages
from utils import get_git_root, log_time

## other component of brightway package
from brightway_loader import BrightwayLoader

root = Path(get_git_root()) 

class GeoLocator:
    ## note at some point we might want to make the geo-variables more specific.
    def __init__(self, brightway_loader: BrightwayLoader, projection=4327):
        self.loader = brightway_loader
        self.G = self.loader.G 
        self.edges_df = self.loader.edges_df
        self.nodes_df = self.loader.nodes_df
        self.projection = projection
        self.setup_background_geo()

    def geolocate(self, subnodes, subedges):
        self.subedges = subedges
        geo_nodes = self.geolocate_nodes(subnodes)
        geo_edges = self.geolocate_edges(subedges, geo_nodes)
        return geo_nodes, geo_edges


    def setup_background_geo(self):
        ## setup the actual shapefiles to map onto
        country_shapefile = root / "Data" / "Shapefiles" / "Natural_earth_countries_all" / "ne_10m_admin_0_countries.shp"
        country_gdf = gpd.read_file(country_shapefile)
        
        fixes = {
            'France': {'ISO_A2': 'FR', 'ISO_A3': 'FRA'},
            'Norway': {'ISO_A2': 'NO', 'ISO_A3': 'NOR'},
            'Kosovo': {'ISO_A2': 'XK', 'ISO_A3': 'XKX'},       
            'Somaliland': {'ISO_A2': 'SO', 'ISO_A3': 'SOL'},   
        }
        for name, codes in fixes.items():
            country_gdf.loc[country_gdf['NAME'] == name, ['ISO_A2', 'ISO_A3']] = codes['ISO_A2'], codes['ISO_A3']

        country_gdf = country_gdf.to_crs(epsg=self.projection)
        self.country_shapefile_frame = country_gdf.copy()

        ## setup the country codes to match on
        excel_file = root / "Data" / "Database-Overview-for-ecoinvent-v3.10_29.04.24.xlsx"
        self.eco_geographies_df = pd.read_excel(excel_file, sheet_name="Geographies")
        self.eco_geographies_df.set_index('Shortname', inplace=True)

    @log_time
    ## TODO at some point, need to make the matching logic more sophisticated to match the ipynb logic
    def geolocate_nodes(self, nodes_df):

        ## match nodes to the ecoinvent geography codes and trim
        geo_nodes = pd.merge(left=nodes_df, right=self.eco_geographies_df, how='left', left_on='location', right_index=True)

        geo_nodes['is_geolocated'] = ~geo_nodes.index.isna()
        geo_nodes.loc[~geo_nodes['is_geolocated'], 'location'] = pd.NA
        geo_nodes['base_name'] = geo_nodes['location'].apply(lambda text: text[:2] if pd.notna(text) else pd.NA)

        ## match the geographic codes for process to the shapefiles
        geo_nodes = pd.merge(
            left=geo_nodes,
            right=self.country_shapefile_frame,
            how='left',
            left_on='base_name',
            right_on='ISO_A2'
        )

        ##  create random points within the geographies, starting with ones that have geometry first
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='geometry', crs=self.projection)
        geo_nodes['is_geolocated'] = ~geo_nodes['geometry'].isna() ## for more accurate filtering later
        geo_nodes.loc[geo_nodes['is_geolocated'], 'random_point'] = geo_nodes.loc[geo_nodes['is_geolocated']].apply(self.generate_random_point, axis=1)
        self.geo_nodes = geo_nodes ## stash for using in find_geoconnected_nodes method
        geo_nodes.loc[~geo_nodes['is_geolocated'], 'random_point'] = geo_nodes.loc[~geo_nodes['is_geolocated']].apply(self.non_geo_point, axis=1)

        # Drop the old geometry column and assign 'random_point' as the new geometry
        geo_nodes = geo_nodes.drop(columns=['geometry'])
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='random_point', crs=self.projection)
        self.geo_nodes = geo_nodes
        return geo_nodes
   
    # @log_time
    def generate_random_point(self, row):
        minx, miny, maxx, maxy = row.geometry.bounds
        while True:
            random_lon = random.uniform(minx, maxx)
            random_lat = random.uniform(miny, maxy)
            point = Point(random_lon, random_lat)

            # Ensure the point is inside the geometry
            if row.geometry.contains(point):
                return point
    
    # @log_time
    ## this is the biggest time loss currently, bc of the loop to find offshore location
    def non_geo_point(self, row): ## this is the biggest time loss. 
        connected_nodes = self.get_connected_geolocated_nodes(row["node"])
        
        if not connected_nodes:
            return self.get_random_offshore_point()

        # Get the first connected geolocated node's position
        nearest_geo_node = connected_nodes[0]
        base_lon, base_lat = nearest_geo_node['Longitude'], nearest_geo_node['Latitude']

        # Helper function to generate a small offshore shift
        def generate_shift():
            shift_lon = np.random.uniform(0.5, 2.0) * (-1 if np.random.rand() > 0.5 else 1)
            shift_lat = np.random.uniform(0.5, 2.0) * (-1 if np.random.rand() > 0.5 else 1)
            return shift_lon, shift_lat

        for _ in range(100):
            shift_lon, shift_lat = generate_shift()
            new_lon = base_lon + shift_lon
            new_lat = base_lat + shift_lat
            
            # Ensure the shifted point is offshore
            if not self.is_on_land(new_lon, new_lat):
                return Point(new_lon, new_lat)

        # If no valid offshore point is found after 100 iterations, return a random offshore point
        return self.get_random_offshore_point()

    def get_random_offshore_point(self):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-85, 85)
        while self.is_on_land(lon, lat):
            lon = np.random.uniform(-180, 180)
            lat = np.random.uniform(-85, 85)
        return Point(lon, lat)
    
    # @log_time
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

    def is_on_land(self, lon, lat):
        """Checks if a point (lon, lat) is on land."""
        point = Point(lon, lat)
        return self.country_shapefile_frame.contains(point).any()

    def geolocate_edges(self, edges, geo_frame):
        node_mapping = geo_frame.set_index("node")["random_point"].to_dict()
        #print(node_mapping)

        # Map the source and target nodes to their points
        edges['source_point'] = edges['source_node'].map(node_mapping)
        edges['target_point'] = edges['target_node'].map(node_mapping)

        #print(edges)

        def make_line(row):
            if row['source_point'] and row['target_point']:
                return LineString([row['source_point'], row['target_point']])
            return None

        # Create the geometry for each edge and filter out None values
        edges['geometry'] = edges.apply(make_line, axis=1)
        edges = edges[edges['geometry'].notna()]

        # Set geometry and CRS to ensure projection consistency
        edges = edges.set_geometry('geometry')
        edges.set_crs(self.projection, inplace=True)

        return edges
