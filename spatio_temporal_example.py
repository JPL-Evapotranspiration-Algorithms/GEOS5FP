import geopandas as gpd

# Load spatio_temporal.csv as a GeoDataFrame
gdf = gpd.read_file('spatio_temporal.csv')

# Print the GeoDataFrame
print(gdf)


