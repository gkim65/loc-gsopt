
import json
import numpy as np
# import random_land_points as rlp

# Brahe Imports
import brahe.data_models as bdm


def gs_json(provider_file):

    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    for sta in stations_json['features']:
        stations.append(bdm.Station(**sta))

    return stations



# # TODO: this may not be needed anymore
# def rand_gs_on_land():

#     # Get a random point on land
#     point = rlp.random_points() # Point is [lon, lat]

#     return bdm.Station(
#             **{
#                 "properties": {
#                     "constraints": bdm.AccessConstraints(elevation_min=0),
#                     "name": "change_this",
#                 },
#                 "type": "Feature",
#                 "geometry": {
#                     "type": "Point",
#                     "coordinates": point # ASK IF THIS IS RIGHT 
#                 },
#             }
#         )

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