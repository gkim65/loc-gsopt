import geopandas as gpd
from shapely.geometry import Point
import json
import matplotlib.pyplot as plt

# Load city JSON
stations_json = json.load(open("teleport_locations.json", 'r'))
cities = [(sta['longitude'], sta['latitude']) for sta in stations_json]

# Create GeoDataFrame in lat/lon
gdf = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lon, lat in cities],
    crs="EPSG:4326"
)

# Approximate buffer in degrees (~50 km)
buffer_radius_deg = 50 / 111
gdf["geometry"] = gdf.buffer(buffer_radius_deg)

# Save shapefile (optional)
gdf.to_file("city_buffers.shp")

# Plot
ax = gdf.plot(edgecolor='blue', facecolor='none', figsize=(10, 10))
plt.title("City Buffers (approx)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig("../figures_final/cityBuffers.pdf", format='pdf', bbox_inches='tight')
plt.show()