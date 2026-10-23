
import json
import numpy as np
import brahe as bh


def gs_json_list(provider_file):
    """
        gs_json_list(provider_file) -> list

    Station coordinates from a GeoJSON FeatureCollection of points (e.g. `data/groundstations/*.json`).

      - `provider_file` — path to the GeoJSON file

    Returns `(lat, lon)` tuples (deg).

    NOTE: the order is latitude first, the reverse of the `[lon, lat]` pairs used elsewhere.
    """
    with open(provider_file, "r") as f:
        data = json.load(f)

    latlon_list = []

    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        latlon_list.append((lat, lon))  # convert to (lat, lon)

    return latlon_list



def gs_json(provider_file):
    """
        gs_json(provider_file) -> list

    Stations from a GeoJSON FeatureCollection of points (e.g. `data/groundstations/*.json`).

      - `provider_file` — path to the GeoJSON file

    Returns `bh.PointLocation` objects in file order.
    """
    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    for feature in stations_json['features']:
        lon, lat = feature["geometry"]["coordinates"][:2]
        stations.append(bh.PointLocation(lon, lat))

    return stations


def teleport_json(provider_file):
    """
        teleport_json(provider_file) -> list

    Load candidate sites from a JSON list of `{"name", "latitude", "longitude", ...}` records.

      - `provider_file` — path to the JSON file

    Returns `bh.PointLocation` objects in file order, each with its list index as its id.
    """
    stations = []

    stations_json = json.load(open(provider_file, 'r'))

    for i, sta in enumerate(stations_json):
        station = bh.PointLocation(sta['longitude'], sta['latitude'])
        station.set_id(i)
        stations.append(station)

    return stations
