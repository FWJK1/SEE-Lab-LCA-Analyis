import plotly.graph_objects as go
import streamlit as st
from interactive_graph import Interactive_Graph
from interactive_graph import log_time


## we build and store the graph
@st.cache_resource
def get_graph():
    return Interactive_Graph()

## we build and store the nodes
@st.cache_resource
def run_graph(_ig):
    _ig.run_lca()
    nodes, edges  = _ig.create_graph_from_lca()
    nodes, geo_nodes, geo_edges = _ig.run(nodes=nodes, edges=edges, levels=0)
    print(len(nodes))
    return nodes, geo_nodes, geo_edges
    
def streamlit_geo(ig, _nodes_df, _edges_df, node=None):
    fig = go.Figure()

    # Plot country boundaries
    x_all, y_all = [], []

    for geometry in ig.country_reference_gdf.geometry:
        if geometry.geom_type == 'Polygon':  # Handle Polygon
            x, y = geometry.exterior.xy
            x_all.append(list(x))
            y_all.append(list(y))
        elif geometry.geom_type == 'MultiPolygon':  # Handle MultiPolygon
            for poly in geometry.geoms:
                x, y = poly.exterior.xy
                x_all.append(list(x))
                y_all.append(list(y))

    for x, y in zip(x_all, y_all):
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            mode='lines',
            line=dict(color='lightgrey', width=2),
            showlegend=False))

    # Add the activity nodes
    activity_types = _nodes_df['activity_type'].unique()
    node_traces = []
    
    # Create checkboxes for activity types
    activity_visibility = {}
    for activity in activity_types:
        activity_visibility[activity] = st.checkbox(f"Show {activity}", value=True)

    for activity in activity_types:
        if activity_visibility[activity]:  # Show the nodes for the selected activity
            filtered_df = _nodes_df[_nodes_df['activity_type'] == activity]
            node_traces.append(go.Scatter(
                x=filtered_df.geometry.x,
                y=filtered_df.geometry.y,
                mode='markers',
                marker=dict(size=10, opacity=0.9),
                name=activity,
                text=filtered_df['name'],
                hoverinfo='text',
                visible=True,
            ))

    for trace in node_traces:
        fig.add_trace(trace)

    # Add the flows
    edge_x, edge_y = [], []
    for line in _edges_df.geometry:
        edge_x.extend([point[0] for point in line.coords] + [None])  # None breaks the line
        edge_y.extend([point[1] for point in line.coords] + [None])

    fig.add_trace(go.Scatter(
        x=edge_x, 
        y=edge_y, 
        mode='lines',
        marker=dict(
            color='lightblue',
            size=1,
            opacity=.8
        ),
        name='Flows'
    ))

    # Display plot using Streamlit
    st.plotly_chart(fig)


@log_time
@st.cache_resource
def precompute_traces(_ig, _nodes_df, _edges_df):
    geo_traces = []
    x_all, y_all = [], []

    for geometry in _ig.country_reference_gdf.geometry:
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

    activity_types = _nodes_df['activity_type'].unique()
    node_traces = {}

    for activity in activity_types:
        filtered_df = _nodes_df[_nodes_df['activity_type'] == activity]
        node_traces[activity] = go.Scatter(
            x=filtered_df.geometry.x,
            y=filtered_df.geometry.y,
            mode='markers',
            marker=dict(size=10, opacity=0.9),
            name=activity,
            text=filtered_df['name'],
            hoverinfo='text',
            visible=True,

        )

    active_node_ids = set(_nodes_df['node'])
    edge_x, edge_y = [], []
    for _, edge in _edges_df.iterrows():
        if edge['source_node'] in active_node_ids and edge['target_node'] in active_node_ids:
            line = edge.geometry
            edge_x.extend([point[0] for point in line.coords] + [None])
            edge_y.extend([point[1] for point in line.coords] + [None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='lightblue', width=.3),
        opacity=.7,
        name='Flows'
    )

    return geo_traces, node_traces, edge_trace

@log_time
def plot_graph(geo_traces, node_traces, edge_trace):
    fig = go.Figure()

    for trace in geo_traces:
        fig.add_trace(trace)

    visible_activities = []
    for activity, trace in node_traces.items():
        show = st.checkbox(f"Show '{activity}' nodes", value=True)
        if show:
            visible_activities.append(activity)
            fig.add_trace(trace)

    show_edges = st.checkbox("Show Flows", value=True)
    if show_edges:
        fig.add_trace(edge_trace)

    fig.update_layout(
        showlegend=True,
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )


    st.plotly_chart(fig, use_container_width=True)

ig = get_graph()
st.title(f"Graph Visualization {ig.foreground_name.split("Backend:")[1]}")
nodes, geo_nodes, geo_edges = run_graph(ig)
geo_traces, activity_traces, edge_trace = precompute_traces(ig, geo_nodes, geo_edges)
plot_graph(geo_traces, activity_traces, edge_trace)
