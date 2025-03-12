## ## python packages
from pathlib import Path

## brightway packages
import bw2io as bi
import bw2data as bd
import bw2calc as bc
from bw2calc.graph_traversal import AssumedDiagonalGraphTraversal

## 3rd party packages 
# import networkx as nx
# import pickle
# import pandas as pd

## homebrew packages
from utils import log_time

class lcaAnalyzer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @log_time
    def run_lca(self):
        wb = bd.Database("Water bottle LCA")
        ef_gwp_key = [m for m in bd.methods if "climate change" in m[1] and "EF" in m[0]].pop()
        if self.verbose:
            print(ef_gwp_key) ## note -- this is key. maybe need to validate somehow? generally make sure methods match foreground data
            print({[act for act in wb][0]: 1})
        my_functional_unit, data_objs, _ = bd.prepare_lca_inputs(
            {[act for act in wb][0]: 1},
            method=ef_gwp_key,
        )
        self.my_lca = bc.LCA(demand=my_functional_unit, data_objs=data_objs)
        self.my_lca.lci()
        self.my_lca.lcia()

    @log_time
    def get_nodes_edges_list_from_lca(self):
        gt = AssumedDiagonalGraphTraversal()
        gt_output = gt.calculate(lca=self.my_lca)
        nodes = gt_output.get('nodes')
        del nodes[-1]
        nodes = [bd.get_activity(id=k)['code'] for k in nodes.keys()]

        edges = gt_output.get('edges')
        del edges[-1]
        edges = [
            (bd.get_activity(edge['from'])['code'], bd.get_activity(edge['to'])['code'])
            for edge in edges
            if edge['from'] != -1 and edge['to'] != -1
        ]

        return nodes, edges
   