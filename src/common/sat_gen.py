import os

import logging
import datetime
import httpx # TODO BRING THIS BACK!

import numpy as np
# import streamlit as st
# import polars as pl

# Brahe Imports
import brahe as bh
import brahe.data_models as bdm

logger = logging.getLogger()

CONSTELLATIONS = sorted(['YAM', 'UMBRA', 'SKYSAT', 'ICEYE', 'FLOCK', 'HAWK', 'CAPELLA', 'LEGION', 'WORLDVIEW', 'GEOEYE',
                  'NUSAT'])

EPHEMERIS_PATH = './data/celestrak_tles.txt' #(pathlib.Path(__file__).parent.parent /  'data/celestrak_tles.txt').absolute()


# Working with Satellite States

# The TLE is likely the most accessible way for you to propagate the satellite state. 
# Brahe has a built-in TLE parser that can be used to load TLEs from lines. 
# However to make it easier, we define a function to create a TLE from orbital elements.
def make_tle(epc0, alt, ecc, inc, raan, argp, M, ndt2=0.0, nddt6=0.0, bstar=0.0, norad_id=99999):
    '''Get a TLE object from the given orbital elements

    Args:
    - epc0 (Epoch): Epoch of the orbital elements / state
    - alt (float): Altitude of the orbit [km]
    - ecc (float): Eccentricity of the orbit
    - inc (float): Inclination of the orbit [deg]
    - raan (float): Right Ascension of the Ascending Node [deg]
    - argp (float): Argument of Perigee [deg]
    - M (float): Mean Anomaly [deg]

    Returns:
    - tle (TLE): TLE object for the given orbital elements
    '''

    alt *= 1e3 # Convert to meters

    # Get semi-major axis
    sma = bh.R_EARTH + alt

    # Get mean motion
    n = bh.mean_motion(sma)/(2*np.pi)*86400

    tle_string = bh.tle_string_from_elements(epc0, np.array([n, ecc, inc, raan, argp, M, ndt2, nddt6, bstar]), norad_id)
    tle = bh.TLE(*tle_string)
    return tle

##########################################

def get_last_modified_time(file_path):
    return os.path.getmtime(file_path)

def get_last_modified_time_as_datetime(file_path):
    return datetime.datetime.fromtimestamp(get_last_modified_time(file_path))

# Celestrak and satellite constellations

# Function for downloading celestrak tles
def get_latest_celestrak_tles(output_dir='./data'):
    CELESTRAK_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=ACTIVE&FORMAT=TLE'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract filename from URL
    filename = os.path.join(output_dir, 'celestrak_tles.txt')

    # Use httpx to get the content from the URL
    response = httpx.get(CELESTRAK_URL, follow_redirects=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a file and write the content
        with open(filename, 'w') as fp:
            fp.write(response.text)
        logger.info(f"Saved latest TLE information to {filename}")
    else:
        logger.error(f"Failed to download TLE data from Celestrak. Status code: {response.status_code}")


def parse_tle_file(filepath):

    # Create an empty list to store parsed TLE records
    tle_records = []

    with open(filepath, 'r') as file:

        # Read all lines from the file
        lines = file.readlines()

        # Iterate over the lines in the file in groups of 3
        i = 0
        while i < len(lines):
            tle_line0 = lines[i].strip()
            tle_line1 = lines[i + 1].strip()
            tle_line2 = lines[i + 2].strip()

            # Get Information
            object_name = tle_line0.rstrip()

            # Extract TLE data
            tle = bh.TLE(tle_line1, tle_line2)

            satcat_id = tle_line1[2:7]
            tle_epoch = tle.epoch.to_datetime(tsys='UTC')
            semi_major_axis = tle.a
            eccentricity = tle.e
            inclination = tle.i
            right_ascension = tle.RAAN
            arg_of_perigee = tle.w
            mean_anomaly = tle.M

            # Append parsed information to the list
            tle_records.append({
                'object_name': object_name,
                'satcat_id': satcat_id,
                'epoch': tle_epoch,
                'altitude': (semi_major_axis - bh.R_EARTH)/1e3,
                'semi_major_axis': semi_major_axis,
                'eccentricity': eccentricity,
                'inclination': inclination,
                'right_ascension': right_ascension,
                'arg_of_perigee': arg_of_perigee,
                'mean_anomaly': mean_anomaly,
                'tle_line0': tle_line0,
                'tle_line1': tle_line1,
                'tle_line2': tle_line2
            })

            # Move to the next 3-line record
            i += 3

    return tle_records

def get_tles(download):

    if not os.path.exists(EPHEMERIS_PATH):
        get_latest_celestrak_tles()

    # Check on time of last update
    last_update = get_last_modified_time_as_datetime(EPHEMERIS_PATH)

    logger.info(f'Last TLE update: {last_update.isoformat()}')


    if download:
        # If the file is older than 1 day, download the latest TLE data
        if (datetime.datetime.now() - last_update).days > 1:
            logger.info(f'TLE data is {(datetime.datetime.now() - last_update).days} days old. Updating...')
            get_latest_celestrak_tles()
        else:
            logger.info(f'TLE data is {(datetime.datetime.now() - last_update).days} days old. TLE data is up to date.')

    # Parse the TLE file and return the records
    return parse_tle_file(EPHEMERIS_PATH)

def satellites_from_constellation(constellation: str, download):

    # Load the TLE data
    tle_data = get_tles(download)

    # Filter the TLE data for the specified constellation
    constellation_tles = [tle for tle in tle_data if constellation.upper() in tle['object_name']]

    # Create a list of Satellite objects from the TLE data
    satellites = [bdm.Spacecraft(
                    id=int(tle['satcat_id']),
                    name=tle['object_name'],
                    line1=tle['tle_line1'],
                    line2=tle['tle_line2']) for tle in constellation_tles]

    return satellites

# @st.cache_resource(ttl=3600*12) # Expire cache every 12 hours
# def get_satcat_df():
#     # Load the TLE data
#     tle_data = get_tles()

#     # Create a DataFrame from the TLE data
#     satcat_df = pl.DataFrame(tle_data, schema={
#         'object_name': str,
#         'satcat_id': str,
#         'epoch': datetime.datetime,
#         'altitude': float,
#         'semi_major_axis': float,
#         'eccentricity': float,
#         'inclination': float,
#         'right_ascension': float,
#         'arg_of_perigee': float,
#         'mean_anomaly': float,
#         'tle_line0': str,
#         'tle_line1': str,
#         'tle_line2': str
#     })

#     return satcat_df


##########################################

# import os
# import pathlib

# import pathlib
# import logging
# import datetime
# import httpx

# import streamlit as st
# import polars as pl
# import brahe as bh

# from gsopt.models import Satellite
# from gsopt.utils import get_last_modified_time_as_datetime


# def satellites_from_constellation(constellation: str, datarate: float = 2.0e9) -> list[Satellite]:

#     # Load the TLE data
#     tle_data = get_tles()

#     # Filter the TLE data for the specified constellation
#     constellation_tles = [tle for tle in tle_data if constellation.upper() in tle['object_name']]

#     # Create a list of Satellite objects from the TLE data
#     satellites = [Satellite(tle['satcat_id'], tle['object_name'], tle['tle_line1'], tle['tle_line2'], datarate=datarate) for tle in constellation_tles]

#     return satellites

# def satellite_from_satcat_id(satcat_id: str, datarate: float = 2.0e9) -> Satellite:

#     # Load the TLE data
#     tle_data = get_tles()

#     # Filter the TLE data for the specified satcat_id
#     tle = next((tle for tle in tle_data if tle['satcat_id'] == str(satcat_id)), None)

#     if tle is None:
#         raise ValueError(f"Satellite with satcat_id {satcat_id} not found in TLE data")

#     # Create a Satellite object from the TLE data
#     satellite = Satellite(tle['satcat_id'], tle['object_name'], tle['tle_line1'], tle['tle_line2'], datarate=datarate)

#     return satellite

##################################################

# def add_constellation(self, name=str):
#     """
#     Add a constellation of satellites to the scenario generator

#     Args:
#         name: str: Name of the constellation to add. Must be one of the following:
#             - YAM
#             - UMBRA
#             - SKYSAT
#             - ICEYE
#             - FLOCK
#             - HAWK
#             - CAPELLA
#             - LEGION
#             - WORLDVIEW
#             - GEOEYE

#     """
#     if name.upper() not in CONSTELLATIONS:
#         raise ValueError(f'Constellation {name} not found in {CONSTELLATIONS}')

#     constellation_sats = self._satcat_df.filter(pl.col('object_name').str.contains(name.upper()))

#     self.satellites.extend(satellites_from_dataframe(constellation_sats))


    # def add_satellite(self, sat_id: str | int):
    #     """
    #     Add a specific satellite to the scenario generator

    #     Args:
    #         sat_id: str: NORAD ID of the satellite to add
    #     """
    #     sat = self._satcat_df.filter(pl.col('satcat_id') == str(sat_id))
    #     self.satellites.extend(satellites_from_dataframe(sat))

    # def add_random_satellites(self, num_satellites: int, alt_range: tuple = (300, 1000)):
    #     """
    #     Add a random selection of satellites to the scenario generator

    #     Args:
    #         num_satellites: int: Number of random satellites to add
    #         sma_range: tuple: Range of altitudes to select random satellites from
    #     """

    #     # Get all satellites with altitudes within the specified range
    #     random_sats = self._satcat_df.filter(pl.col('altitude').is_between(alt_range[0], alt_range[1]))

    #     # Get all unique satellite NORAD IDs
    #     sat_ids = list(sorted(random_sats['satcat_id'].unique().to_list()))

    #     # Randomly select a subset of the satellite NORAD IDs
    #     selected_sat_ids = self._rng.sample(sat_ids, num_satellites)

    #     # Filter the satellite catalog DataFrame to only include the selected satellites
    #     selected_sats = self._satcat_df.filter(pl.col('satcat_id').is_in(selected_sat_ids))

    #     # Add the selected satellites to the scenario generator
    #     self.satellites.extend(satellites_from_dataframe(selected_sats))

        