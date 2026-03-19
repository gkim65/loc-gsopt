
import json
import numpy as np
import brahe as bh


def gs_json_list(provider_file):

    # load file
    with open(provider_file, "r") as f:
        data = json.load(f)

    latlon_list = []

    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        latlon_list.append((lat, lon))  # convert to (lat, lon)

    return latlon_list



def gs_json(provider_file):

    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    for feature in stations_json['features']:
        lon, lat = feature["geometry"]["coordinates"][:2]
        stations.append(bh.PointLocation(lon, lat))

    return stations


def teleport_json(provider_file):
    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    stations.append(bh.PointLocation(sta['longitude'], sta['latitude']))

    return stations