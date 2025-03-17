## python packages 
from pathlib import Path
import random
import os
from concurrent.futures import ThreadPoolExecutor


## 3rd party packages 
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString

## homebrew packages
from lca_graph_viz.helpers import get_git_root, log_time
from .brightway_loader import BrightwayLoader
from .offshore_gen import OffshorePointGenerator

root = Path(get_git_root()) 

class GeoLocator:
    ## note at some point we might want to make the geo-variables more specific.
    def __init__(self, brightway_loader: BrightwayLoader, projection=4327, geomode="offshore"):
        self.loader = brightway_loader
        self.G = self.loader.G 
        self.edges_df = self.loader.edges_df
        self.nodes_df = self.loader.nodes_df
        self.projection = projection
        self.geomode = geomode
        self.setup_background_geo()

    def geolocate(self, subnodes, subedges):
        self.subedges = subedges
        geo_nodes = self.geolocate_nodes(subnodes)
        geo_edges = self.geolocate_edges(subedges, geo_nodes)
        return geo_nodes, geo_edges


    def setup_background_geo(self):
        ## setup the country shapefiles to map onto
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

        ## get the states shapefile as well
        state_shapefile = root / "Data" / "Shapefiles" / "Natural_earth_states_provinces_all" / "ne_10m_admin_1_states_provinces.shp"
        self.states_shapefile_frame = gpd.read_file(state_shapefile).to_crs(epsg=self.projection)

        ## setup the ecoinvent encoding to match on
        excel_file = root / "Data" / "Database-Overview-for-ecoinvent-v3.10_29.04.24.xlsx"
        self.eco_geographies_df = pd.read_excel(excel_file, sheet_name="Geographies")
        self.eco_geographies_df.set_index('Shortname', inplace=True)

        ## setup the indexing frames to move from state level matching to country level matching
        index_file = root / "Data" / "indexing_csvs" /"country-and-continent-codes-list.csv"
        self.indexing_codes = pd.read_csv(index_file)



    def encode_nodes(self, nodes_df):
        ## match nodes to the ecoinvent geography codes and trim
        encoded_nodes = pd.merge(left=nodes_df, right=self.eco_geographies_df, how='left', left_on='location', right_index=True)
        encoded_nodes['is_geolocated'] = ~encoded_nodes.index.isna()

        if self.geomode == "antarctica":
            encoded_nodes.loc[~encoded_nodes['is_geolocated'], 'location'] = "AQ"
            encoded_nodes.loc[encoded_nodes['location'] =='GLO', 'location'] = "AQ"
            encoded_nodes.loc[encoded_nodes['location'] =='RoW', 'location'] = "AQ"

        elif self.geomode == "offshore":
            encoded_nodes.loc[encoded_nodes['location'] =='GLO', 'location'] = pd.NA
            encoded_nodes.loc[~encoded_nodes['is_geolocated'], 'location'] = pd.NA
        encoded_nodes['base_name'] = encoded_nodes['location'].apply(lambda text: text[:2] if pd.notna(text) else pd.NA)
        return encoded_nodes


    def match_encoded_nodes(self, encoded_nodes):
            geo_nodes = pd.merge(
                left=encoded_nodes,
                right=self.country_shapefile_frame,
                how='left',
                left_on='base_name',
                right_on='ISO_A2'
            )
            return geo_nodes

        ## potentially useful later
        # matched_states = pd.merge(
        #     left = geo_nodes,
        #     right = self.states_shapefile_frame,
        #     how='inner',
        #     left_on='location',
        #     right_on="iso_3166_2"
        # )
        # matched_states = matched_states

        # geo_nodes = pd.merge(
        #     left=geo_nodes,
        #     right=matched_states,
        #     how='left',
        #     left_on='ISO_A2',
        #     right_on='Two_Letter_Country_Code'
        # )

    @log_time
    def geolocate_nodes(self, nodes_df):
        encoded_nodes = self.encode_nodes(nodes_df)
        geo_nodes = self.match_encoded_nodes(encoded_nodes)

        ##  create random points within the geographies, starting with ones that have geometry first
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='geometry', crs=self.projection)
        geo_nodes['is_geolocated'] = geo_nodes['geometry'].notna()  ## for more accurate filtering later
        geo_nodes.loc[geo_nodes['is_geolocated'], 'random_point'] = geo_nodes.loc[geo_nodes['is_geolocated']].apply(self.bounded_random_point, axis=1)
        self.geo_nodes = geo_nodes ## stash for using in find_geoconnected_nodes method

        self.offshore_gen = OffshorePointGenerator(self.country_shapefile_frame, self.geo_nodes, self.subedges)
        rows_needed = geo_nodes.loc[~geo_nodes['is_geolocated']]
        offshore_results = self.parallel_offshore(rows_needed)
        print(f"Rows needing geolocation: {len(rows_needed)}")
        print(f"Offshore points generated: {len(offshore_results)}")        

        geo_nodes.loc[rows_needed.index, 'random_point'] = offshore_results
        geo_nodes.drop(columns='geometry', inplace=True)
        
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='random_point', crs=self.projection)
        self.geo_nodes = geo_nodes
        return geo_nodes
    
    def offshore_row(self, row):
        return self.offshore_gen.non_geo_point(row)
   
    def parallel_offshore(self, df, num_workers= max(1, os.cpu_count() - 4)):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.offshore_row, [row for _, row in df.iterrows()]))
        return results

    # @log_time
    def bounded_random_point(self, row):
        minx, miny, maxx, maxy = row.geometry.bounds
        while True:
            random_lon = random.uniform(minx, maxx)
            random_lat = random.uniform(miny, maxy)
            point = Point(random_lon, random_lat)

            # Ensure the point is inside the geometry
            if row.geometry.contains(point):
                return point

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
