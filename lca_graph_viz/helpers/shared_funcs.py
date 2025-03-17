#standard python libs
import random 

#3rd party libs
import networkx

def get_random_node(graph):
    node = random.choice(list(graph.nodes))
    sphere = graph.nodes[node]['sphere']
    if sphere == 'biosphere':
        return get_random_node(graph=graph)
    else:
        return node
 