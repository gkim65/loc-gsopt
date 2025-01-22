import os.path

# Brahe Imports
import brahe as bh
import brahe.access.access as ba
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


import math

def load_earth_data(filename):
    # Here we can download the latest Earth orientation data and load it.

    if not os.path.exists(filename):
        # Uncomment this line ONCE the data has been downloaded. Recomment it once it has been downloaded.
        print("Downloading latest earth orientation data...")
        if not os.path.exists("data"):
            os.makedirs("data")
        bh.utils.download_iers_bulletin_ab("./data")
        print("Download complete")

    # Load the latest Earth Orientation Data
    print("Loading the latest Earth Orientation Data")
    bh.EOP.load(filename)


def gap_times(contacts, epc_start, epc_end):

    gap_time_list = [(contacts[i].t_end,contacts[i+1].t_start) for i in range(len(contacts)-1)]
    
    if len(contacts) != 0:
        if epc_start != contacts[0].t_start:
            gap_time_list.append((epc_start,contacts[0].t_start))
        if epc_end != contacts[-1].t_end:
            gap_time_list.append((contacts[-1].t_end,epc_end))
    
    return gap_time_list

def compute_all_gaps_contacts(satellites,ground_stations,epc_start,epc_end, plot):

    if plot:
        fig, ax = fig, ax =  plt.subplots(figsize=(20,7))
    contacts_all = []
    gaps_all = []

    # for multiprocessing
    tasks = []

    for sat,id_sat in zip(satellites,range(len(satellites))):
        for gs,id_gs in zip(ground_stations,range(len(ground_stations))):
            tasks.append([sat,gs,epc_start,epc_end])

    mpctx = mp.get_context('fork')
    with mpctx.Pool(mp.cpu_count()) as pool:

        results = pool.starmap(ba.find_location_accesses, tasks)

        for contacts in results:
            gaps = gap_times(contacts, epc_start.to_datetime(),epc_end.to_datetime())

            if plot:
                ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (id_sat+0.05 + 10*(id_gs-1), 0.85),facecolors='tab:blue')
                ax.broken_barh([(gaps[i][0],gaps[i][1]-gaps[i][0]) for i in range(len(gaps))], (id_sat+0.05 + 10*(id_gs-1), 0.85),facecolors='tab:green')

            for contact in contacts:
                contacts_all.append(contact)
            for gap in gaps:
                gaps_all.append(gap)
    
    if plot:
        plt.ylabel("different sat ids")
        plt.xlabel("time period over a day")
        plt.legend()
        plt.show()

    gaps_seconds = []
    for gap_tuple in gaps_all:
        gaps_seconds.append((gap_tuple[1]-gap_tuple[0]).total_seconds())

    return contacts_all, gaps_all, gaps_seconds

def compute_all_gaps_contacts_no_mp(satellites,ground_stations,epc_start,epc_end, plot):

    if plot:
        fig, ax = fig, ax =  plt.subplots(figsize=(20,7))
    contacts_all = []
    gaps_all = []

    for sat,id_sat in zip(satellites,range(len(satellites))):
        for gs,id_gs in zip(ground_stations,range(len(ground_stations))):
            contacts = ba.find_location_accesses(sat,gs,epc_start,epc_end)
            gaps = gap_times(contacts, epc_start.to_datetime(),epc_end.to_datetime())

            if plot:
                ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (id_sat+0.05 + 10*(id_gs-1), 0.85),facecolors='tab:blue')
                ax.broken_barh([(gaps[i][0],gaps[i][1]-gaps[i][0]) for i in range(len(gaps))], (id_sat+0.05 + 10*(id_gs-1), 0.85),facecolors='tab:green')

            for contact in contacts:
                contacts_all.append(contact)
            for gap in gaps:
                gaps_all.append(gap)
    
    if plot:
        plt.ylabel("different sat ids")
        plt.xlabel("time period over a day")
        plt.legend()
        plt.show()

    gaps_seconds = []
    for gap_tuple in gaps_all:
        gaps_seconds.append((gap_tuple[1]-gap_tuple[0]).total_seconds())

    return contacts_all, gaps_all, gaps_seconds

def compute_earth_interior_angle(ele=0.0, alt=525):
    '''This function computes the earth interior angle for a given elevation angle and altitude.
    This is the angle between the satellite and the observer on the Earth's surface viewing the satellite
    at the given elevation angle. This is more useful for plotting than the look angle, since it
    can used alonside Earth's Radius to draw a circle around the subsatellite point to get
    the view cone of observers on the Earth's surface that would be able to see the satellite.


    Args:
    - ele (float): Elevation angle of the satellite [deg]
    - alt (float): Altitude of the satellite [km]

    '''
    ele = ele * math.pi / 180.0

    rho = math.asin(bh.R_EARTH/(bh.R_EARTH + alt * 1e3))

    eta = math.asin(math.cos(ele)*math.sin(rho))
    lam = math.pi/2.0 - eta - ele

    return lam