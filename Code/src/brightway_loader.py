## python packages
from pathlib import Path

## brightway packages
import bw2io as bi
import bw2data as bd
import bw2calc as bc

## 3rd party packages 
import networkx as nx
import pickle
import pandas as pd

## homebrew packages
from utils import get_git_root, log_time

root = Path(get_git_root()) 

class BrightwayLoader:
    _instance = None

    ## ensures there is only one brightway_loader at a time, so we don't duplicate the effort
    def __new__(cls, graph_data=None, project="SEE_LAB",
                technosphere="ecoinvent-3.9.1-cutoff", biosphere="ecoinvent-3.9.1-biosphere", 
                foreground="Water bottle LCA"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(graph_data, project, technosphere, biosphere, foreground)
        return cls._instance

    @log_time
    def _initialize(self, graph_data, project, technosphere, biosphere, foreground):
        bd.projects.set_current(project)
        self.technosphere = bd.Database(technosphere)
        self.biosphere = bd.Database(biosphere)
        self.foreground = bd.Database(foreground)
        self.foreground_name = foreground

        # Set default graph_data path if none is provided
        self.graph_path = Path(graph_data) if graph_data else root / "Data" / "saved_networks" / "eco_3.9.1" / "eco_3-9-1_graph"

        if self.graph_path.exists():
            try:
                self.load_graph()
                self.load_frames()
            except (FileNotFoundError, ValueError) as e:
                print(f"Error '{e}', rebuilding graph and frames.")
                self.rebuild_graph_and_frames()
        else:
            self.rebuild_graph_and_frames()

    def rebuild_graph_and_frames(self):
        self.build_graph()
        self.save_graph()
        self.build_frames()
        self.save_frames()

    @log_time
    def build_graph(self):
        self.G = nx.DiGraph()
        spheres = {
            "technosphere": self.technosphere,
            "biosphere": self.biosphere,
            "foreground": self.foreground
        }

        for sphere, acts in spheres.items():  # Add nodes
            for act in acts:
                self.add_node(act, sphere)

        for _, acts in spheres.items():  # Add edges
            for act in acts:
                for exc in act.exchanges():
                    self.add_edge(exc)

        self.validate_graph()

    def validate_graph(self):
        results = {
            "invalid_nodes": [n for n in self.G.nodes if isinstance(n, tuple)],
            "invalid_edges": [(u, v) for u, v in self.G.edges if u not in self.G.nodes or v not in self.G.nodes],
            "isolates": [node for node in nx.isolates(self.G) if self.G.nodes[node]['sphere'] != 'biosphere']
        }

        for key, value in results.items():
            if value:
                raise ValueError(f"{key.replace('_', ' ').capitalize()}: {value}")

    def add_edge(self, exchange):
        exchange = exchange.as_dict()
        input_code = exchange.get('input')
        output_code = exchange.get('output')

        if isinstance(input_code, tuple):
            input_code = input_code[1]
        if isinstance(output_code, tuple):
            output_code = output_code[1]

        self.G.add_edge(input_code, output_code, **exchange)

    def add_node(self, node, sphere):
        node = node.as_dict()
        node['sphere'] = sphere
        node['activity type'] = node.get("activity type", sphere)
        node = {k.replace(" ", "_"): v for k, v in node.items()}
        self.G.add_node(node['code'], **node)

    ### Saving and Loading ###
    def load_graph(self):
        with open(self.graph_path, "rb") as f:
            self.G = pickle.load(f)
        self.validate_graph()

    def save_graph(self):
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.G, f)

    @log_time
    def save_frames(self):
        self.nodes_df.to_csv(self.graph_path.with_name(self.graph_path.name + "_nodes.csv"), index=False)
        self.edges_df.to_csv(self.graph_path.with_name(self.graph_path.name + "_edges.csv"), index=False)

    @log_time
    def load_frames(self):
        self.nodes_df = pd.read_csv(self.graph_path.with_name(self.graph_path.name + "_nodes.csv"), low_memory=False)
        self.edges_df = pd.read_csv(self.graph_path.with_name(self.graph_path.name + "_edges.csv"), low_memory=False)

    @log_time
    def build_frames(self):
        ## build nodes
        df_nodes = pd.DataFrame([{'node': n, **d} for n, d in self.G.nodes(data=True)])

        ## add relevant graph metrics and then map to pandas 
        between_dict = nx.betweenness_centrality(self.G, k=min(500, len(self.G)))
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())
        df_nodes['betweenness_centrality'] = df_nodes['node'].map(between_dict)
        df_nodes['in_degree'] = df_nodes['node'].map(in_degrees)
        df_nodes['out_degree'] = df_nodes['node'].map(out_degrees)

        df_edges = pd.DataFrame([{'source_node': u, 'target_node': v, **d} for u, v, d in self.G.edges(data=True)])

        self.nodes_df = df_nodes
        self.edges_df = df_edges

if __name__ == "__main__":
    bload = BrightwayLoader()
    print(bload.edges_df)
    print(bload.nodes_df)
