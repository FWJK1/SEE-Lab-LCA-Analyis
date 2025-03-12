
## python packages
from pathlib import Path
from itertools import chain


## 3rd party packages 
import networkx as nx

## homebrew packages
from utils import log_time
from brightway_loader import BrightwayLoader

class SubgraphProcessor:
    def __init__(self, brightway_loader: BrightwayLoader):
        self.loader = brightway_loader
        self.G = self.loader.G 
        self.nodes_df = self.loader.nodes_df
        self.edges_df = self.loader.edges_df

    @log_time
    def build_subgraph(self, nodelist, edges=[],  levels=1):
        ## note that this isn't really optimized at present (just uses a big set)
        ## and will probably have to be improved at some point if we want lots of levels.. 
        visited = set(nodelist)
        visited_edges = set(edges)
        parents = visited
        count = 0
        while count < levels:
            next_parents = set()
            for parent in parents:
                incoming_nodes = set(self.G.predecessors(parent))
                outgoing_nodes = set(self.G.successors(parent))
                outgoing_nodes = {node for node in outgoing_nodes if node in visited}
                neighbors = incoming_nodes.union(outgoing_nodes)
                for node in neighbors:
                    visited_edges.add((node, parent))
                next_parents.update(neighbors)
            visited.update(next_parents)
            parents = next_parents
            count +=1
        
        subgraph=self.G.subgraph(visited).copy()
        subgraph.add_edges_from(visited_edges)
        self.subgraph = subgraph
        return subgraph
    
    def frames_from_subgraph(self, subgraph):
        node_frame = self.nodes_df[self.nodes_df['node'].isin(subgraph.nodes())].copy()
        source_nodes = [u for u, v in subgraph.edges() if u != v]
        target_nodes = [v for u, v in subgraph.edges() if u != v]
        edge_frame = self.edges_df[self.edges_df['source_node'].isin(source_nodes) & self.edges_df['target_node'].isin(target_nodes)].copy()
        return node_frame, edge_frame

