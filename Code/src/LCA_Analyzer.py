## python packages
from pathlib import Path

## brightway packages
import bw2io as bi
import bw2data as bd
import bw2calc as bc
from bw2calc.graph_traversal import AssumedDiagonalGraphTraversal

## homebrew packages
from utils import log_time

class lcaAnalyzer:
    """
    Class for performing LCA (Life Cycle Assessment) analysis using Brightway. It handles running the LCA, 
    extracting relevant results, and providing nodes and edges from the LCA traversal.

    Attributes:
        verbose (bool): Flag to control verbosity of print statements.
        my_lca (LCA): LCA instance used for the analysis.
    """
    
    def __init__(self, verbose=True):
        """
        Initializes the LCA analyzer with an optional verbosity flag.

        Args:
            verbose (bool): Whether to print additional information during the process.
        """
        self.verbose = verbose

    @log_time
    def run_lca(self):
        """
        Runs the LCA calculation using the "Water bottle LCA" database. 
        It prepares the functional unit and the necessary data for the LCA, 
        then computes the life cycle inventory (LCI) and life cycle impact assessment (LCIA).

        Raises:
            ValueError: If the key for the climate change method is not found or there is an issue with the method.
        """
        wb = bd.Database("Water bottle LCA")
        
        # Find the method related to climate change (GWP)
        ef_gwp_key = [m for m in bd.methods if "climate change" in m[1] and "EF" in m[0]].pop()

        if self.verbose:
            print(ef_gwp_key)  # Key for the method, may need validation
            print({[act for act in wb][0]: 1})  # Initial functional unit

        # Prepare the functional unit and data objects for the LCA
        my_functional_unit, data_objs, _ = bd.prepare_lca_inputs(
            {[act for act in wb][0]: 1},  # Using the first activity as the functional unit
            method=ef_gwp_key,  # Using the chosen method
        )
        
        # Create the LCA object and run the calculations
        self.my_lca = bc.LCA(demand=my_functional_unit, data_objs=data_objs)
        self.my_lca.lci()  # Life Cycle Inventory
        self.my_lca.lcia()  # Life Cycle Impact Assessment

    @log_time
    def get_nodes_edges_list_from_lca(self):
        """
        Retrieves the list of nodes and edges from the LCA analysis, based on a graph traversal.

        Uses the AssumedDiagonalGraphTraversal method to extract nodes and edges. 
        Returns a list of node codes and edge connections between nodes.

        Returns:
            tuple: A tuple containing:
                - nodes (list): List of activity codes for nodes.
                - edges (list): List of tuples representing edges between activity codes.
        """
        gt = AssumedDiagonalGraphTraversal()  # Initialize the graph traversal
        gt_output = gt.calculate(lca=self.my_lca)  # Perform the traversal
        
        # Get nodes and remove the last element (which is the root node)
        nodes = gt_output.get('nodes')
        del nodes[-1]
        nodes = [bd.get_activity(id=k)['code'] for k in nodes.keys()]  # Map to activity codes

        # Get edges and remove the last element
        edges = gt_output.get('edges')
        del edges[-1]
        edges = [
            (bd.get_activity(edge['from'])['code'], bd.get_activity(edge['to'])['code'])
            for edge in edges
            if edge['from'] != -1 and edge['to'] != -1  # Exclude invalid edges
        ]

        return nodes, edges
