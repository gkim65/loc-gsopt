import geopandas as gpd
from shapely.geometry import Point
import json
import os
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(DATA_DIR), "figures_final")

# Load city JSON
stations_json = json.load(open(os.path.join(DATA_DIR, "teleport_locations.json"), 'r'))
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
gdf.to_file(os.path.join(DATA_DIR, "city_buffers.shp"))

# Plot
ax = gdf.plot(edgecolor='blue', facecolor='none', figsize=(10, 10))
plt.title("City Buffers (approx)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "cityBuffers.pdf"), format='pdf', bbox_inches='tight')
plt.show()