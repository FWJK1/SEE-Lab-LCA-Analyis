import bw2io as bi
import bw2data as bd
import bw2calc as bc
from bw2calc.graph_traversal import AssumedDiagonalGraphTraversal
import git
import logging

import networkx as nx
from networkx.algorithms import bipartite

import random
from collections import defaultdict
import json
import re
import pickle

import pandas as pd
import geopandas as gpd
import numpy as np

from datetime import datetime
from functools import wraps
import time

from matplotlib import pyplot as plt 
import seaborn as sns
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely.geometry import LineString
import plotly.graph_objects as go
import streamlit as st
import sys



class SQLFilter(logging.Filter):
    def __init__(self, exclude_patterns=None):
        # Initialize with a list of patterns to exclude (default: empty list)
        self.exclude_patterns = exclude_patterns or []

    def filter(self, record):
        # Get the message part of the log
        message = record.getMessage()

        # Check if the message (query) starts with any of the exclude_patterns
        return not any(message.startswith(pattern) for pattern in self.exclude_patterns)

# List of patterns to filter out
exclude_patterns = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP"]

    
logging.basicConfig(
    level=logging.INFO,  # Change to INFO, WARNING, etc. for different verbosity
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console (stdout)
        logging.FileHandler("app.log")      # Output to a file
    ],
)

def get_git_root():
    repo = git.Repo(search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")
root = get_git_root()

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

def printline():
    logging.info("---" * 50)

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        printline()
        logging.info(f"Started {func.__name__} at {get_current_time()}")
        result = func(*args, **kwargs)
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        logging.info(f"Finished {func.__name__} at {get_current_time()}")
        logging.info(f"Total time elapsed: {elapsed_time:.4f} seconds")  # Print elapsed time
        printline()
        logging.info("\n")
        return result
    return wrapper


def print_dict(dictionary):
    logging.info(json.dumps(dictionary, indent=4))
                     

class Interactive_Graph:
    @log_time
    def __init__(self, graph_data=f"{root}/Data/saved_networks/eco_3.9.1/eco_3-9-1_graph", project="SEE_LAB",
                 technosphere="ecoinvent-3.9.1-cutoff", biosphere="ecoinvent-3.9.1-biosphere", verbose=0,
                 foreground="Water bottle LCA"):
        bd.projects.set_current(project) 
        self.technosphere = bd.Database(technosphere)
        self.biosphere = bd.Database(biosphere)
        self.foreground = bd.Database(foreground)
        self.foreground_name = foreground
        self.verbose = verbose
      
        if graph_data:
            try:
                self.load_graph(graph_data)
                try: 
                    self.load_frames(graph_data)
                except Exception as e:
                    logging.info(f"Exception '{e}', so rebuild the frames ")
                    self.build_frames()
                    self.save_frames(graph_data)
            except Exception as e:
                logging.info(f"Exception '{e}', so rebuild the graph and frames")
                self.build_graph()
                self.save_graph(graph_data)
                self.build_frames()
                self.save_frames(graph_data)
        else: 
            self.build_graph()
            self.build_frames()

        self.setup_background_geo()


    ### building the main graph for initial setup### 
    @log_time
    def build_graph(self):
        self.G = nx.DiGraph()
        ## build the nodes, building all of them first
        for act in self.technosphere: 
            self.add_node(node=act, sphere="technosphere")
        for act in self.biosphere: 
            self.add_node(node=act, sphere="biosphere")
        for act in self.foreground:
            self.add_node(node=act, sphere='foreground')

        ## build the edges
        for act in self.technosphere:
            for exc in act.exchanges():
                self.add_edge(exc)
        for act in self.biosphere:
            for exc in act.exchanges():
                self.add_edge(exc)
        for act in self.foreground:
            for exc in act.exchanges():
                self.add_edge(exc)
        self.validate_graph()

    def validate_graph(self):
        results = {
            "invalid_nodes": [n for n in self.G.nodes if isinstance(n, tuple)],
            "invalid_edges": [(u, v) for u, v in self.G.edges if u not in self.G.nodes or v not in self.G.nodes],
            # "isolates": list(nx.isolates(self.G)),  ## we leave this out because lots of bioshere nodes are isolated (not sure why)
        }
        for key, value in results.items():
            if value:
                raise ValueError(f"{key.replace('_', ' ').capitalize()}: {value}")
            
    def add_edge(self, exchange):
        exchange = exchange.as_dict()
        # some input and outputs are listed as tuples
        input_code = exchange['input'][1] if isinstance(exchange['input'], tuple) else exchange['input']
        output_code = exchange['output'][1] if isinstance(exchange['output'], tuple) else exchange['output']
        self.G.add_edge(input_code, output_code, **exchange)

    def add_node(self, node, sphere):
        node = node.as_dict()
        node['sphere'] = sphere
        code = node['code']
        node['activity type'] = node.get("activity type", sphere)

        node = {k.replace(" ", "_"): v for k, v in node.items()}
        if isinstance(code, tuple):
            logging.info(code)
        self.G.add_node(node['code'], **node)


    ### Saving and Loading Frames and Graphs ###
    def load_graph(self, graph_data):
        with open(graph_data, "rb") as f: 
            self.G = pickle.load(f)
        self.validate_graph()

    def save_graph(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    def save_frames(self, path):
        self.nodes_df.to_csv(path+"_nodes_dataframe.csv")
        self.edges_df.to_csv(path+"_edges_dataframe.csv")
        # self.nodes_df.to_parquet(path+"_nodes_dataframe", engine='pyarrow')
        # self.edges_df.to_parquet(path+"_edges_dataframe", engine='pyarrow')

    def load_frames(self, path):
        self.nodes_df = pd.read_csv(path+"_nodes_dataframe.csv", low_memory=False)
        self.edges_df = pd.read_csv(path+"_edges_dataframe.csv", low_memory=False)


    ## saving and loading geoframes ##
    def save_geo_frame(self, path):
        self.nodes_geo_df.to_parquet(path+"_geo_nodes_dataframe", engine='fastparquet')

    def load_geo_frame(self, path):
        self.nodes_geo_df = gpd.read_parquet(path+"_geo_nodes_dataframe", engine='fastparquet')


    ### getting rough statistics on graph ###

    ### building subgraphs ### 
    @log_time
    def build_subgraph_from_nodelist(self, nodes, levels=1, edges=[]):
        ## note that this isn't really optimized at present (just uses a big set)
        ## and will probably have to be improved at some point if we want lots of levels.
        visited = set(nodes)
        visited_edges = set(edges)
        parents = visited
        count = 0
        while count < levels:
            next_parents = set()
            for parent in parents:
                incoming_nodes = list(self.G.predecessors(parent))
                outgoing_nodes = list(self.G.successors(parent))
                outgoing_nodes = [node for node in outgoing_nodes if node in visited]
                neighbors = set(incoming_nodes).union(outgoing_nodes)
                for node in neighbors:
                    visited_edges.add((node, parent))
                next_parents.update(neighbors)
            visited.update(next_parents)
            parents = next_parents
            count +=1
        
        subgraph=self.G.subgraph(visited).copy()
        subgraph.add_edges_from(visited_edges)
        return subgraph

    def get_random_node(self, graph=None):
        if graph is None:
            graph = self.G
        a = random.choice(list(graph.nodes))
        sphere = graph.nodes[a]['sphere']
        if sphere == 'biosphere':
            return self.get_random_node(graph=graph)
        else:
            return a


    def setup_background_geo(self):
    
        country_gdf = gpd.read_file(f"{root}/Data/Shapefiles/Natural_earth_countries_all/ne_10m_admin_0_countries.shp")
        fixes = {
            'France': {'ISO_A2': 'FR', 'ISO_A3': 'FRA'},
            'Norway': {'ISO_A2': 'NO', 'ISO_A3': 'NOR'},
            'Kosovo': {'ISO_A2': 'XK', 'ISO_A3': 'XKX'},       
            'Somaliland': {'ISO_A2': 'SO', 'ISO_A3': 'SOL'},   
        }
        for name, codes in fixes.items():
            country_gdf.loc[country_gdf['NAME'] == name, ['ISO_A2', 'ISO_A3']] = codes['ISO_A2'], codes['ISO_A3']

        country_gdf.to_crs(epsg=4327)
        self.country_reference_gdf = country_gdf.copy()

    @log_time
    def build_frames(self, graph=None):
        graph = graph or self.G
        
        # Create the nodes and edges DataFrames
        df_nodes = pd.DataFrame([
            {'node': n, **d} for n, d in graph.nodes(data=True)
        ])
        df_nodes['in_degree'] = df_nodes['node'].apply(lambda x: graph.in_degree(x))
        df_nodes['out_degree'] = df_nodes['node'].apply(lambda x: graph.out_degree(x))

        df_edges = pd.DataFrame([
            {'source_node': u, 'target_node': v, **d} for u, v, d in graph.edges(data=True)
        ])
        
        if graph is self.G:
            self.nodes_df = df_nodes
            self.edges_df = df_edges
        else:
            return df_nodes, df_edges


    def build_node_geoframe(self, nodes=None):
        if nodes is None:
            nodes = self.nodes_df

        ## add geotags to the node frame
        eco_geographies_df = pd.read_excel(f"{root}/Data/Database-Overview-for-ecoinvent-v3.10_29.04.24.xlsx", sheet_name="Geographies")
        geo_nodes = pd.merge(
            left=nodes,
            right=eco_geographies_df,
            how='left', ## at some point we have to make this more sophisticated (eg, better location detail
            left_on='location',
            right_on='Shortname'
        )
        geo_nodes.loc[geo_nodes['Shortname'].isna(), 'location'] = 'AQ'
        ## helper func
        def get_basename(text):
            text = str(text)
            if len(text) == 2:
                return text
            match = re.search(r"..-", text) 
            return match.group(0)[:2] if match else None
        
        geo_nodes.drop(columns='Shortname', inplace=True)
        geo_nodes['base_name'] = geo_nodes['location'].apply(get_basename)

        ## add shapefile info to the node frame
        geo_nodes = pd.merge(
            left = geo_nodes,
            right=self.country_reference_gdf,
            how='inner', ## at some point we have to make this outer and make it more sophisticated 
            left_on='base_name',
            right_on='ISO_A2'
        )

        # print(geo_nodes)

        ## add random points to the geo_frame
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='geometry', crs="EPSG:4327")
        geo_nodes['random_point'] = geo_nodes.geometry.apply(self.generate_random_point)
        geo_nodes = gpd.GeoDataFrame(geo_nodes, geometry='random_point')


        if nodes is self.nodes_df:
            self.nodes_geo_df = geo_nodes.copy()
        else:
            return geo_nodes

    def build_edge_geoframe(self, edges, geo_frame):

        geo_frame = geo_frame.to_crs(self.country_reference_gdf.crs)
        node_mapping = geo_frame.set_index("node")["random_point"].to_dict() ## so that point to where the nodes are going to be shown

        def make_line(row):
            try:
                return LineString([node_mapping[row["source_node"]], node_mapping[row["target_node"]]])
            except KeyError:
                return None
            
        edges['geometry'] = edges.apply(make_line, axis=1)
        edges = edges[edges['geometry'].notna()]
        edges = edges.set_geometry('geometry')
        edges.set_crs("EPSG:4327", inplace=True)
        logging.debug(edges.columns)
        return edges



    def generate_random_point(self, geometry):
        # Get the bounding box of the geometry
        minx, miny, maxx, maxy = geometry.bounds
        
        # Generate random coordinates within the bounding box
        random_lon = random.uniform(minx, maxx)
        random_lat = random.uniform(miny, maxy)
        
        # Generate a point
        point = Point(random_lon, random_lat)
        
        # Ensure the point is inside the geometry
        while not geometry.contains(point):
            random_lon = random.uniform(minx, maxx)
            random_lat = random.uniform(miny, maxy)
            point = Point(random_lon, random_lat)
        return point
    
    

    ### Run LCA ### 
    @log_time
    def run_lca(self):
        wb = bd.Database("Water bottle LCA")
        ef_gwp_key = [m for m in bd.methods if "climate change" in m[1] and "EF" in m[0]].pop()
        logging.info(ef_gwp_key) ## note -- this is key. maybe need to validate somehow? generally make sure methods match foreground data
        logging.info({[act for act in wb][0]: 1})
        my_functional_unit, data_objs, _ = bd.prepare_lca_inputs(
            {[act for act in wb][0]: 1},
            method=ef_gwp_key,
        )
        self.my_lca = bc.LCA(demand=my_functional_unit, data_objs=data_objs)
        self.my_lca.lci()
        self.my_lca.lcia()

    @log_time
    def create_graph_from_lca(self):
        gt = AssumedDiagonalGraphTraversal()
        gt_output = gt.calculate(lca=self.my_lca)
        nodes = gt_output.get('nodes')
        del nodes[-1]
        nodes = [bd.get_activity(id=k)['code'] for k in nodes.keys()]
        print(len(nodes))

        edges = gt_output.get('edges')
        del edges[-1]
        edges = [
            (bd.get_activity(edge['from'])['code'], bd.get_activity(edge['to'])['code'])
            for edge in edges
            if edge['from'] != -1 and edge['to'] != -1
        ]

        return nodes, edges
   

    ### Visualizations ###
    @log_time
    def plot_geo_graph(self, nodes_gdf=None, node=None, edges_gdf=None):
        if nodes_gdf is None:
            nodes_gdf = self.nodes_geo_df
        if node is None:
            node = self.get_random_node()

        logging.info(self.country_reference_gdf.total_bounds)
        logging.info(nodes_gdf.total_bounds)
        logging.info(edges_gdf.total_bounds)


        fig, ax = plt.subplots(figsize=(15, 15))
        self.country_reference_gdf.plot(ax=ax, edgecolor='gray', linewidth=0.2, facecolor='none')
        nodes_gdf.plot(ax=ax, markersize=3, color="blue", alpha=.6)
        edges_gdf.plot(ax=ax, color='red', linewidth=.4, alpha=0.2, label="Edges")
        plt.show()



    ## non geo-network plot ## 
    @log_time
    def plot_subgraph(self, subgraph, node=None):
        if node is None:
            node = self.get_random_node(subgraph)

        palette = sns.color_palette("Set2", 4)
        edges_to_draw = [(u, v) for u, v in subgraph.edges() if u != v]

        node_colors = [
            palette[0] if n == node else
            palette[1]
            for n in subgraph.nodes()
        ]

        edge_colors = [
            palette[2] if v == node else
            palette[1] if u == node else
            palette[3]
            for u, v in edges_to_draw
        ]

        labels = {
            n: data.get('name', '') if n == node else ''
            for n, data in subgraph.nodes(data=True)
        }

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph)  # or nx.kamada_kaway_layout(subgraph) for better layout sometimes
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors)
        nx.draw_networkx_edges(subgraph, pos, edgelist=edges_to_draw, edge_color=edge_colors)
        plt.title("Subgraph Visualization")
        plt.show()


    @log_time
    def plotly_geo(self, nodes_df, edges_df, node=None):
        fig = go.Figure()

        # Plot country boundaries
        x_all, y_all = [], []

        for geometry in self.country_reference_gdf.geometry:
            if geometry.geom_type == 'Polygon':  # Handle Polygon
                x, y = geometry.exterior.xy
                x_all.append(list(x))
                y_all.append(list(y))
            elif geometry.geom_type == 'MultiPolygon':  # Handle MultiPolygon
                for poly in geometry.geoms:
                    x, y = poly.exterior.xy
                    x_all.append(list(x))
                    y_all.append(list(y))

        for x, y in zip(x_all, y_all):
            fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode='lines',
                line=dict(color='lightgrey', width=2),
                showlegend=False))


        ## add the activity nodes
        activity_types = nodes_df['activity_type'].unique()
        node_traces = []
        for activity in activity_types:
            filtered_df = nodes_df[nodes_df['activity_type'] == activity]
            node_traces.append(go.Scatter(
                x=filtered_df.geometry.x,
                y=filtered_df.geometry.y,
                mode='markers',
                marker=dict(size=10, opacity=0.9),
                name=activity,
                text=filtered_df['name'],
                hoverinfo='text',
                visible=True
            ))


        for trace in node_traces:
            fig.add_trace(trace)


        ## add the flows
        flow_trace_index = len(node_traces) 
        edge_x, edge_y = [], []
        for line in edges_df.geometry:
            edge_x.extend([point[0] for point in line.coords] + [None])  # None breaks the line
            edge_y.extend([point[1] for point in line.coords] + [None])

        fig.add_trace(go.Scatter(
            x=edge_x, 
            y=edge_y, 
            mode='lines',
            marker=dict(
                color='lightblue',
                size=1,
                opacity=.8
            ),
            name='Flows'
            ))


        # Add dropdown menu to toggle visibility
        buttons = []
        for i, activity in enumerate(activity_types):
            buttons.append(
                dict(
                    label=f"Toggle {activity}",
                    method='restyle',
                    args=['visible', [True]],
                    args2=['visible', [False]],
                )
            )
            buttons.append(
            dict(
                label='Toggle Flows',
                method='restyle',
                args=['visible', [True]],
                args2=['visible', [False]],
                traces=[flow_trace_index]
            )
        )



        ## add the buttons to the menu
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="down",
                buttons=buttons,
                showactive=False,  # Important for toggles
                x=0.1,
                y=1.15
            )],
            title="Toggle Visibility by Activity Type"
        )

    def run(self, save=None, nodes=None, show_subgraph=False, levels=1, edges=[]):
        if nodes is None:
            nodes = [self.get_random_node()]
        subgraph = self.build_subgraph_from_nodelist(nodes, levels=levels, edges=edges)
        if show_subgraph: 
            self.plot_subgraph(subgraph, nodes[0])
        subnodes, subedges = self.build_frames(subgraph)
        geo_nodes = self.build_node_geoframe(subnodes)
        geo_edges = self.build_edge_geoframe(subedges, geo_nodes)
        return nodes, geo_nodes, geo_edges

if __name__ == "__main__":
    IG = Interactive_Graph()
    IG.run_lca()
    nodes, edges = IG.create_graph_from_lca()
    nodes, geo_nodes, geo_edges =IG.run(nodes=nodes, edges=edges, show_subgraph=True, levels=0)

    
## to do

# 1. add ability to generate subgraph from lca
# 2. make the basic plotly work
# 3. start adding plotly filters:
    # node type
    # node degree
    # 