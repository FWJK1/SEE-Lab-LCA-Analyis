## 3rd party packages
import plotly.graph_objects as go
import networkx as nx
import seaborn as sns
import matplotlib.colors as mcolors

class NxPlotter:
    def __init__(self, subgraph, nodes):
        self.G = subgraph
        self.layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "spectral": nx.spectral_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout
        }
        activity_types  = nodes['activity_type'].unique().tolist()
        palette = sns.color_palette("Set2", len(activity_types))
        self.activity_color_map = {
            activity: mcolors.to_hex(palette[i])  # Convert each Seaborn RGB to hex
            for i, activity in enumerate(activity_types)
        }

    def create_figure(self, filtered_nodes, layout, node_size_attr=None, show_edges=True, **kwargs):
        fig = go.Figure()
        subgraph, pos = self.get_pos(filtered_nodes, layout)
        for trace in self.compute_node_traces(subgraph, pos, filtered_nodes['activity_type'].unique().tolist(), node_size_attr=node_size_attr):
            fig.add_trace(trace)
        
        if show_edges:
            fig.add_trace(self.compute_edge_trace(subgraph, pos))

        fig.update_layout(
            autosize=True,  
            margin=dict(l=10, r=10, t=10, b=10), 
        )
        return fig

    def get_pos(self, filtered_nodes, layout):
        subgraph = self.G.subgraph(filtered_nodes['node'].tolist())
        layout_func = self.layout_funcs.get(layout, nx.spring_layout)
        pos = layout_func(subgraph)
        return subgraph, pos
    
    def compute_edge_trace(self, subgraph, pos):
        edge_x = []
        edge_y = []

        for source, target, _ in subgraph.edges(data=True):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])  # None creates a break in the line for separate edges
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='grey'),
            opacity=0.8,
            hoverinfo='none',
            mode='lines',
            name="Flows"
        )
        return edge_trace
    
    def compute_node_traces(self, subgraph, pos, activity_types, node_size_attr=None):
        # Create node traces, one for each unique activity_type
        node_traces = []

        for activity in activity_types:
            # Filter nodes by the current activity type
            nodes_of_type = [node for node, data in subgraph.nodes(data=True) if data.get("activity_type") == activity]

            node_x = []
            node_y = []
            node_sizes = []
            node_colors = []
            node_hover_text = []

            for node in nodes_of_type:
                data = subgraph.nodes[node]
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                node_sizes.append(data.get(node_size_attr, 15) if node_size_attr else 10)
                node_colors.append(self.activity_color_map[activity])  # Use the mapped color for the activity
                node_hover_text.append(f"{data.get("location", 'Biosphere')} : {data.get("name", '')}")

            # Create a trace for the current activity type
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                name=f"{activity}",  # Add the activity type as the legend entry
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                ),
                text=node_hover_text,  # Set hover text to display the node's "name"
                hoverinfo='text'
            )
            node_traces.append(node_trace)

        return node_traces




