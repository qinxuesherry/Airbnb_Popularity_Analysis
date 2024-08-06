import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import branca.colormap as cm

df1 = pd.read_csv('Data/listings_detail.csv')

# Load the GeoJSON data for the regions
regions_gdf = gpd.read_file("Data/Bangkok-districts.geojson")

# Convert df1 to a GeoDataFrame
points_gdf = gpd.GeoDataFrame(
    df1,
    geometry=gpd.points_from_xy(df1.longitude, df1.latitude),
    crs=regions_gdf.crs
)

# Perform a spatial join to count points within each region
points_in_regions = gpd.sjoin(points_gdf, regions_gdf, how='inner', op='within')
region_counts = points_in_regions.groupby('index_right').size()
regions_gdf['point_count'] = regions_gdf.index.map(region_counts).fillna(0)

# Create a color scale
max_count = regions_gdf['point_count'].max()
min_count = regions_gdf['point_count'].min()
color_scale = cm.linear.YlOrRd_09.scale(min_count, max_count)

# Apply color scale to the regions
regions_gdf['color'] = regions_gdf['point_count'].apply(color_scale)

# Create a base map
centroid = regions_gdf.geometry.centroid.iloc[0]
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=6)

# Add the regions to the map with colors based on point counts
folium.GeoJson(
    regions_gdf,
    name='Bangkok Regions',
    style_function=lambda feature: {
        'fillColor': feature['properties']['color'],
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(fields=['point_count'],
                                  aliases=['Number of Points:']),
).add_to(m)

# Add the color scale to the map
color_scale.caption = 'Number of Points per Region'
color_scale.add_to(m)

# Display the map
output_file = 'Data/airbnb_map.html'    
m.save(output_file)
