import plotly.graph_objects as go


## change this and change piped in logic to shift to the edges as a list of traces, too.
## we want to be able to filter edges to whats not there    
class GeoPlotter:
    def __init__(self, geo_traces, node_traces, edge_trace):
        """Initialize the plotter with precomputed traces."""
        self.geo_traces = geo_traces
        self.node_traces = node_traces
        self.edge_trace = edge_trace

    def create_figure(self, visible_activities, show_edges):
        """Builds and returns a Plotly figure based on selected data."""
        fig = go.Figure()

        # Add country boundaries
        for trace in self.geo_traces:
            fig.add_trace(trace)

        # Add node traces for selected activities
        for activity in visible_activities:
            if activity in self.node_traces:
                fig.add_trace(self.node_traces[activity])

        # Add edges if selected
        if show_edges:
            fig.add_trace(self.edge_trace)

        fig.update_layout(
            showlegend=True,
            height=650,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig
