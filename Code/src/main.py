
## software subcomponents
from brightway_loader import BrightwayLoader
from non_geo_viz import NonGeoVisualizer
from LCA_Analyzer import lcaAnalyzer
from subgraph_gen import SubgraphProcessor
from geo_locator import GeoLocator
from shared_funcs import get_random_node


## homebrew utilities
from utils import log_time


@log_time
def main():
    # Initialize Brightway loader
    bload = BrightwayLoader()

    # Run LCA analysis
    lca_analyzer = lcaAnalyzer()
    lca_analyzer.run_lca()
    nodes, edges = lca_analyzer.get_nodes_edges_list_from_lca()

        ## Build Subgraph and create frames 

    sp= SubgraphProcessor(bload)
    subgraph = sp.build_subgraph(nodes, edges=edges, levels=0)
    subnodes, subedges = sp.frames_from_subgraph(subgraph)

    # ## Build Subgraph and create frames 
    # sp = SubgraphProcessor(bload)
    # subgraph = sp.build_subgraph([get_random_node(bload.G)])
    # subnodes, subedges = sp.frames_from_subgraph(subgraph)



    ## geolocate data
    geol = GeoLocator(bload)
    subnodes, subedges = geol.geolocate(subnodes, subedges)


    # Visualize resulte non-geo
    visualizer = NonGeoVisualizer(bload)
    visualizer.plot_subgraph(subgraph)


if __name__ == "__main__":
    main()
