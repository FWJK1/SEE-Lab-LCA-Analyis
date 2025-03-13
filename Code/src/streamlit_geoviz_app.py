import streamlit as st
from streamlit_geoviz_state_manager import initialize_state

# Initialize session state variables
initialize_state()

# Retrieve objects from session state
bload = st.session_state.bload
plotter = st.session_state.plotter

# --- Streamlit UI --- #
st.title(f"Graph Visualization {bload.foreground_name}")

# User selections
visible_activities = [activity for activity in plotter.node_traces.keys() if st.checkbox(f"Show '{activity}' nodes", value=True)]
show_edges = st.checkbox("Show Flows", value=True)

# Generate and display the figure
fig = plotter.create_figure(visible_activities=visible_activities, show_edges=show_edges)
st.plotly_chart(fig, use_container_width=True)
