## python packages
from pathlib import Path
import random

## brightway packages
import bw2io as bi
import bw2data as bd
import bw2calc as bc

## 3rd party packages 
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt

## homebrew packages
from utils import get_git_root, log_time

## other component of brightway package
from brightway_loader import BrightwayLoader
from shared_funcs import get_random_node


root = Path(get_git_root()) 

class NonGeoVisualizer():
    def __init__(self, brightway_loader: BrightwayLoader):
        self.loader = brightway_loader
        self.G = self.loader.G



    @log_time
    def plot_subgraph(self, subgraph, node=None, savepath=None):
        if node is None:
            node = get_random_node(subgraph)

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
