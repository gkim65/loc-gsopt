import os
from common.paths import repo_path

import logging
import datetime
import httpx 

import numpy as np

# Brahe Imports
import brahe as bh

logger = logging.getLogger()

# Commercial constellations whose satellite names appear in the CelesTrak TLE snapshot
CONSTELLATIONS = sorted(['YAM', 'UMBRA', 'SKYSAT', 'ICEYE', 'FLOCK', 'HAWK', 'CAPELLA', 'LEGION', 'WORLDVIEW', 'GEOEYE',
                  'NUSAT'])

# TLE snapshot used for every run; see the README for its date
EPHEMERIS_PATH = repo_path('data/celestrak_tles.txt')


def make_tle(epc0, alt, ecc, inc, raan, argp, M, norad_id=99999):
    """
        make_tle(epc0, alt, ecc, inc, raan, argp, M, norad_id=99999) -> (str, str)

    Two-line element set for a satellite with the given mean orbital elements.

      - `epc0` — element epoch (`bh.Epoch`)
      - `alt` — altitude above the equatorial radius (km)
      - `ecc` — eccentricity
      - `inc` — inclination (deg)
      - `raan` — right ascension of the ascending node (deg)
      - `argp` — argument of perigee (deg)
      - `M` — mean anomaly (deg)
      - `norad_id` — catalog number written into the TLE

    Returns TLE lines 1 and 2.
    """

    alt *= 1e3 # Convert to meters

    # Get semi-major axis
    sma = bh.R_EARTH + alt

    line1, line2 = bh.keplerian_elements_to_tle(epc0, np.array([sma, ecc, inc, raan, argp, M]), str(norad_id))

    return line1, line2

def get_last_modified_time(file_path):
    """get_last_modified_time(file_path) -> float: modification time of `file_path` (s since the epoch)."""
    return os.path.getmtime(file_path)

def get_last_modified_time_as_datetime(file_path):
    """get_last_modified_time_as_datetime(file_path) -> datetime.datetime: modification time of `file_path`, local time."""
    return datetime.datetime.fromtimestamp(get_last_modified_time(file_path))


def get_latest_celestrak_tles(output_dir=repo_path('data')):
    """
        get_latest_celestrak_tles(output_dir=repo_path('data')) -> None

    Download CelesTrak's current active-satellite TLEs to `output_dir/celestrak_tles.txt`.

      - `output_dir` — directory to write to, created if missing

    NOTE: this overwrites the committed snapshot, which changes every result that uses it.
    """
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
    """
        parse_tle_file(filepath) -> list

    Parse a three-line TLE file (name line, line 1, line 2).

      - `filepath` — path to the TLE file

    Returns one dict per satellite: `object_name`, `satcat_id`, `epoch` (`bh.Epoch`),
    `semi_major_axis` (m), `altitude` (km above the equatorial radius), `eccentricity`,
    `inclination`, `right_ascension`, `arg_of_perigee`, `mean_anomaly` (deg) and the three raw lines.
    """
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
            epoch, tle = bh.keplerian_elements_from_tle(tle_line1, tle_line2)
            

            satcat_id = tle_line1[2:7]
            tle_epoch = epoch
            semi_major_axis = tle[0]
            eccentricity = tle[1]
            inclination = tle[2]
            right_ascension = tle[3]
            arg_of_perigee = tle[4]
            mean_anomaly = tle[5]

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
    """
        get_tles(download) -> list

    Parsed records from `EPHEMERIS_PATH`, downloading the file if it is missing.

      - `download` — also re-download when the file is more than a day old

    Returns the records from `parse_tle_file`.

    NOTE: `download=True` (`debug.txtUpdate`) replaces the snapshot the paper used.
    """
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
    """
        satellites_from_constellation(constellation, download) -> list

    SGP4 propagators for every satellite whose TLE name contains `constellation`.

      - `constellation` — case-insensitive name fragment, e.g. "CAPELLA" or "ICEYE"
      - `download` — passed to `get_tles`

    Returns unpropagated `bh.SGPPropagator` objects in TLE-file order; callers take the first
    `problem.sat_num`.
    """
    # Load the TLE data
    tle_data = get_tles(download)

    # Filter the TLE data for the specified constellation
    constellation_tles = [tle for tle in tle_data if constellation.upper() in tle['object_name']]

    # Create a list of Satellite objects from the TLE data
    satellites = [bh.SGPPropagator.from_tle(tle['tle_line1'], tle['tle_line2']) for tle in constellation_tles]

    return satellites

def make_walker_constellation(
    epoch=bh.Epoch(2024, 5, 20, 0, 0, 0),
    altitude_km=550,
    eccentricity=0.001,
    inclination=90,
    num_planes=6,
    sats_per_plane=4,
    phase=1,
    argp=0.0,
    norad_start=10000,
    star = True,
):
    """
        make_walker_constellation(epoch, altitude_km=550, eccentricity=0.001, inclination=90, num_planes=6, sats_per_plane=4, phase=1, argp=0.0, norad_start=10000, star=True) -> list

    Walker constellation of near-circular orbits at one altitude and inclination.

      - `epoch` — element epoch (`bh.Epoch`)
      - `altitude_km` — altitude (km)
      - `eccentricity` — eccentricity
      - `inclination` — inclination (deg)
      - `num_planes`, `sats_per_plane` — constellation size
      - `phase` — Walker phasing factor between adjacent planes
      - `argp` — argument of perigee (deg)
      - `norad_start` — catalog number of the first satellite
      - `star` — spread planes over 180 deg of RAAN (Walker-Star) instead of 360 deg (Walker-Delta)

    Returns `bh.SGPPropagator` objects, plane by plane.
    """
    total_sats = num_planes * sats_per_plane
    plane_spacing = 180 if star else 360  # Key difference between Star/Delta
    constellation = []

    for p in range(num_planes):
        raan = (plane_spacing * p / num_planes)
        for s in range(sats_per_plane):
            mean_anomaly = ( (360.0 / sats_per_plane) * s + \
                 (phase * p * 360.0 / total_sats) ) % 360.0
            norad_id = norad_start + p * sats_per_plane + s
            
            line1, line2 = make_tle(
                epc0=epoch,
                alt=altitude_km,
                ecc=eccentricity,
                inc=inclination,
                raan=raan,
                argp=argp,
                M=mean_anomaly,
                norad_id=norad_id
            )

            sat_name = f"Sat_{p}_{s}"
            sat_obj = bh.SGPPropagator.from_tle(line1, line2)
            constellation.append(sat_obj)

    return constellation
