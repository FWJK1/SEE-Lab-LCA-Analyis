import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
import geopandas as gpd

projection_mapping = {
    4326: 'mercator',  # WGS 84 (latitude/longitude), commonly used for web maps
    3857: 'mercator',  # Web Mercator
    3395: 'orthographic',  # Popular for global maps
    2163: 'albers usa',  # US-based Albers projection
}

class HeatPlotter:
    def __init__(self, geol):
        self.shapefile = geol.country_shapefile_frame
        self.projection = geol.projection
        self.plotly_projection = projection_mapping.get(self.projection, 'mercator') 
        
    def merge_nodes(self, filtered_nodes):
        counts = filtered_nodes['location'].value_counts().reset_index()
        counts.columns = ['location', 'count']
        df =  pd.merge(
            left=counts,
            right=self.shapefile,
            how='right',
            left_on="location",
            right_on="ISO_A2"
        )
        df['count'] = df['count'].fillna(0)
        gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=self.projection)
        return gdf
    
    def plot_heatmap(self, gdf, ax):
        ax.axis("off")
        cm = mpl.colormaps.get_cmap("YlOrRd")
        norm = mpl.colors.Normalize(vmin=max(gdf['count'].min(), 1), vmax=max(gdf['count'].max(), 50))
        gdf.plot(column='count', norm=norm, cmap=cm, ax=ax,
               edgecolor='black',
                linewidth = 0.3 )
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm._A = []  
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.0)
        cbar.set_label('Process Count')  


    def create_figure(self, filtered_nodes, **kwargs):
        # Merge the filtered nodes with the geo_nodes for geometries
        gdf = self.merge_nodes(filtered_nodes)
        fig, ax = plt.subplots(1, figsize=(14, 16))
        self.plot_heatmap(gdf, ax)
        plt.tight_layout() 
        return fig
