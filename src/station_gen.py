
import json
import numpy as np
# from global_land_mask import globe

# Brahe Imports
import brahe.data_models as bdm


def gs_json(provider_file):

    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    for sta in stations_json['features']:
        stations.append(bdm.Station(**sta))

    return stations



# TODO: this may not be needed anymore
def rand_gs_on_land():

    on_water = True
    lat, long = 0,0
    counter = 0
    while on_water:
        # generate random location coordinates
        lat = np.random.uniform(-90, 90)
        long = np.random.uniform(-180, 180)

        on_water = globe.is_ocean(lat,long)
        counter += 1

    return bdm.Station(
            **{
                "properties": {
                    "constraints": bdm.AccessConstraints(elevation_min=0),
                    "name": "change_this",
                },
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [long, lat] # ASK IF THIS IS RIGHT 
                },
            }
        ), counter, lat, long

def return_bdm_gs(lon,lat):
    return bdm.Station(
            **{
                "properties": {
                    "constraints": bdm.AccessConstraints(elevation_min=0),
                    "name": "change_this",
                },
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat] # ASK IF THIS IS RIGHT 
                },
            }
        )