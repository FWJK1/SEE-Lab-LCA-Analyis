import plotly.graph_objects as go


## change this and change piped in logic to shift to the edges as a list of traces, too.
## we want to be able to filter edges to whats not there    
class GeoPlotter:
    def __init__(self, geol, node_df, edge_df):
        """Default constructor: Compute all traces."""
        self.geo_traces = self.compute_geo_traces(geol)
        self.node_traces = self.compute_node_traces(node_df)
        self.edge_traces = self.compute_edge_traces(edge_df)

    @classmethod
    def precompute_load(cls, geo_traces, node_traces, edge_traces):
        """Alternative constructor: Initialize with precomputed traces."""
        instance = cls.__new__(cls)  
        instance.geo_traces = geo_traces
        instance.node_traces = node_traces
        instance.edge_traces = edge_traces
        return instance


    ## precomputing methods ## 
    def compute_geo_traces(self, geol):
        """
        Compute the static geographic boundaries.
        Requires a GeoLocator object as the geol var to get the shapefile.
        """
        geo_traces = []
        x_all, y_all = [], []

        for geometry in geol.country_shapefile_frame.geometry:
            if geometry.geom_type == 'Polygon':
                x, y = geometry.exterior.xy
                x_all.append(list(x))
                y_all.append(list(y))
            elif geometry.geom_type == 'MultiPolygon':
                for poly in geometry.geoms:
                    x, y = poly.exterior.xy
                    x_all.append(list(x))
                    y_all.append(list(y))

        for x, y in zip(x_all, y_all):
            geo_traces.append(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='lightgrey', width=2),
                showlegend=False
            ))
        return geo_traces
    
    def compute_node_traces(self, nodes_df):
        node_traces = {}
        activity_types = nodes_df['activity_type'].unique()
        for activity in activity_types:
            filtered_df = nodes_df[nodes_df['activity_type'] == activity]
            node_traces[activity] = go.Scatter(
                x=filtered_df.geometry.x,
                y=filtered_df.geometry.y,
                mode='markers',
                marker=dict(size=10, opacity=0.9),
                name=activity,
                text=filtered_df['location'] + ' : ' + filtered_df['name'],
                hoverinfo='text',
                customdata=filtered_df['node']
            )
        return node_traces
    
    def compute_edge_traces(self, edges_df):
        """
        Calculate all the edge traces, to be filtered later
        """
        edge_traces = {}

        for _, edge in edges_df.iterrows():
            line = edge.geometry
            edge_x = [point[0] for point in line.coords] + [None]
            edge_y = [point[1] for point in line.coords] + [None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='lightblue', width=.3),
                opacity=.7,
                name=f"{edge['source_node']} -> {edge['target_node']}",  # Using source and target node IDs as the name
                showlegend=False 
            )
            id = (edge['source_node'], edge['target_node'])
            edge_traces[id] = edge_trace
        return edge_traces
    
        ## maybe consider, for code simplicity, using INVISIBLE activites and filtering OUT those.
    def filter_nodes(self, visible_activities):
        active_nodes = set()
        traces = []
        def update_nodes_and_traces(trace):
            traces.append(trace)
            active_nodes.update(trace.customdata)

        if visible_activities is not None:
            for activity in visible_activities:
                trace = self.node_traces[activity]
                update_nodes_and_traces(trace)
        else:
            for trace in self.node_traces.values():
                update_nodes_and_traces(trace)
        return active_nodes, traces

    def filter_edges(self, active_nodes):
        return [
            trace for (source_node, target_node), trace in
            self.edge_traces.items() 
            if source_node in active_nodes and target_node in active_nodes
            ]


    def create_figure(self, visible_activities=None, show_edges=True):
        """Builds and returns a Plotly figure based on selected data."""
        fig = go.Figure()

        # Add country boundaries
        for trace in self.geo_traces:
            fig.add_trace(trace)

        # filter and add node traces
        active_nodes, node_traces = self.filter_nodes(visible_activities)
        fig.add_traces(node_traces)

        # filter and add edge_traces
        if show_edges:
            fig.add_traces(self.filter_edges(active_nodes))

        # fig.update_layout(
        #     showlegend=True,
        #     height=650,
        #     margin=dict(l=0, r=0, t=40, b=0),
        # )

        return fig
