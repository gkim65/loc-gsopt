import os.path

# Brahe Imports
import brahe as bh
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from itertools import chain
matplotlib.use('Agg')


import math

from pyomo.environ import *
from itertools import combinations, groupby
import pyomo.kernel as pk


def xyz_to_latlon(xyz):
    xyz = xyz / np.linalg.norm(xyz)
    x, y, z = xyz
    lat_rad = np.arcsin(z)  # sin(lat) = z
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.array([lat_deg, lon_deg])


def latlon_to_xyz(lat_deg, lon_deg):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.array([x, y, z])


def contactExclusion(contacts, cfg):
    """
    Solves the contact exclusion problem using a pairwise overlap constraint approach.
    """

    model = ConcreteModel()

    # to call contacts in the correct order 
    contacts_order = {
        str(contact.id):i
        for i, contact in enumerate(contacts)
    }


    # Decision variables: x[c] = 1 if contact c is selected, 0 otherwise
    model.x = Var(range(len(contacts)), within=Binary)


    if cfg.problem.objective == "maximize_num_contacts":
        # Objective: Maximize total scheduled duration
        model.obj = Objective(
            expr=sum(model.x[i] for i in range(len(contacts))), 
            sense=maximize
        )

 
    if cfg.problem.objective == "data_downlink":
        # Objective: Maximize total scheduled duration
        model.obj = Objective(
            expr=sum(model.x[i] * contact.duration 
                    for i, contact in enumerate(contacts)), 
            sense=maximize
        )

    # Constraint list to ensure no overlapping contacts
    model.constraints = ConstraintList()

    # Ensure that no two overlapping contacts for the same satellite are selected
    contacts_sorted_by_satellite = sorted(contacts, key=lambda cn: cn.satellite_id)

    for sat_id, satellite_contacts in groupby(contacts_sorted_by_satellite, lambda cn: cn.satellite_id):

        # Convert groupby object to a sorted list
        satellite_contacts = sorted(list(satellite_contacts), key=lambda cn: cn.t_start)

        # Test all pairs of contacts to check for overlap
        for x, y in combinations(satellite_contacts, 2):
            if x.t_start <= y.t_end and y.t_start <= x.t_end:
                model.constraints.add(model.x[contacts_order[str(x.id)]] + model.x[contacts_order[str(y.id)]] <= 1)

    # miminum duration constraint
    for cn in contacts:
        if cn.duration <= 180:
            # Force all contacts with duration less than the minimum to be zero
            model.constraints.add(model.x[contacts_order[str(cn.id)]] == 0)

    # Solve the model
    solver = SolverFactory('cbc')
    solver.solve(model)

    # Extract selected contacts
    selected_contacts = [contact for i, contact in enumerate(contacts) if value(model.x[i]) > 0.75]
    contact_times_seconds = [contact.t_end - contact.t_start for contact in selected_contacts]

    return selected_contacts, contact_times_seconds


def compute_contact_times(satellites, ground_stations, epc_start, epc_end, min_elevation_deg=10.0):
    """
    Compute contact times using Brahe's parallel propagation + batch access.
    """

    # Step 1: Propagate all satellites in parallel
    bh.par_propagate_to(satellites, epc_end)

    # Step 2: Define access constraints
    constraint = bh.ElevationConstraint(min_elevation_deg)

    # Step 3: Compute access windows for all satellites at once
    windows = bh.location_accesses(
        ground_stations,
        satellites,
        epc_start,
        epc_end,
        constraint
    )

    # Step 4: Flatten results and compute durations
    all_contacts = []
    contact_times_seconds = []

    for c in windows:
        all_contacts.append(c)
        contact_times_seconds.append(c.t_end - c.t_start)

    return all_contacts, contact_times_seconds

# mostly just for plotting
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



####################################### NEED TO CHECK ####################################### 
# TODO: haven't checked 
# for full data downlink / full number of contact times maximizing this
def compute_contact_times_old(satellites,ground_stations,epc_start,epc_end, plot,title="contact_times_chart.png"):

    # plot boolean used for any plotting function
    if plot:
        fig, ax = fig, ax =  plt.subplots(figsize=(20,15))
        len_gs = len(ground_stations)+2

    # setup of output saving
    all_contacts = []

    # Group tasks by satellite for multiprocessing
    grouped_tasks = []
    for sat in satellites:
        sat_tasks = []
        for gs in ground_stations:
            sat_tasks.append([sat, gs, epc_start, epc_end])
        grouped_tasks.append(sat_tasks)

    # Complete the multi processing of getting location accesses for all satellite tasks
    mpctx = mp.get_context('fork')
    with mpctx.Pool(mp.cpu_count()) as pool:
        all_results = []
        for sat_tasks in grouped_tasks:
            sat_results = pool.starmap(ba.find_location_accesses, sat_tasks)
            all_results.append(sat_results)

    # For all contacts on satellites, condense into one large list
    # Also useful for plotting
    for id_gs,sat_result in enumerate(all_results):
        all_contacts_sat = []
        for id_sat,contacts in enumerate(sat_result):
            
            # Plotting per satellite and gs contact windows
            if plot:
                ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (id_sat+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:blue',label = "contacts")

            # Condense contacts
            for contact in contacts:
                all_contacts_sat.append(contact)

                # I don't think I need this:
                # if plot:
                #     ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (-1+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:purple')

        # Saving all satellite outputs into full constellation outputs
        all_contacts.append(all_contacts_sat)
    
    all_contacts_flattened = list(chain.from_iterable(all_contacts))

    # Plotting
    if plot:
        # ax.broken_barh([(all_gap_times[i][0],all_gap_times[i][1]-all_gap_times[i][0]) for i in range(len(all_gap_times))], (-1.9, 0.85),facecolors='tab:green',label = "gaptimes")
        plt.ylabel("Different Satellite groups, With Contact / Gap Times per Ground station")
        plt.xlabel("Time period over a day")
        # plt.legend()
        plt.savefig(title)

    contact_times_seconds = [(contact.t_end - contact.t_start).total_seconds() for contact in all_contacts_flattened]
        
    return all_contacts, contact_times_seconds




# More for getting gap times between one specific contact task, like one satellite to gs
# Using for plotting for now, may not need this for computations
def gap_times(contacts, epc_start, epc_end):
    gap_time_list = [(contacts[i].t_end,contacts[i+1].t_start) for i in range(len(contacts)-1)]
    
    if len(contacts) != 0:
        if epc_start < contacts[0].t_start:
            gap_time_list.append((epc_start,contacts[0].t_start))
        if epc_end > contacts[-1].t_end:
            gap_time_list.append((contacts[-1].t_end,epc_end))
    
    return gap_time_list


# Condensing gap times when there are overlapping contacts (Specifically should be used on per satellite basis)
def gap_times_condense(contacts_all, epc_start, epc_end):
    all_gap_times = []
    gaps_seconds = []

    sorted_contacts = sorted(contacts_all, key=lambda c: c.t_start)
    current_t_end = sorted_contacts[0].t_end

    for contact in sorted_contacts[1:]:  # Start from the second contact
        if current_t_end < contact.t_end and current_t_end > contact.t_start:
            # Overlapping contact → Extend `current_t_end`
            current_t_end = contact.t_end
        elif current_t_end < contact.t_start:
            # Found a real gap → Append its duration
            all_gap_times.append((current_t_end,contact.t_start))
            gaps_seconds.append((contact.t_start - current_t_end).total_seconds())
            current_t_end = contact.t_end  # Move to next contact's end time
    
    # Add the first and last gap time at ends of the simulation
    if sorted_contacts:
        if epc_start < sorted_contacts[0].t_start:
            all_gap_times.append((epc_start,sorted_contacts[0].t_start))
            gaps_seconds.append((sorted_contacts[0].t_start - epc_start).total_seconds())
        if epc_end > sorted_contacts[-1].t_end:
            all_gap_times.append((sorted_contacts[-1].t_end,epc_end))
            gaps_seconds.append((epc_end - sorted_contacts[-1].t_end).total_seconds())

    return all_gap_times,gaps_seconds


# for the correct contacts for gap times per satellite
def compute_gaps_per_sat(satellites,ground_stations,epc_start,epc_end, plot,title="gap_times_chart.png"):

    # plot boolean used for any plotting function
    if plot:
        fig, ax = fig, ax =  plt.subplots(figsize=(20,15))
        len_gs = len(ground_stations)+2

    # setup of output saving
    all_contacts = []
    all_gaps = []
    all_gaps_secs = []

    # Group tasks by satellite for multiprocessing
    grouped_tasks = []
    for sat in satellites:
        sat_tasks = []
        for gs in ground_stations:
            sat_tasks.append([sat, gs, epc_start, epc_end])
        grouped_tasks.append(sat_tasks)

    # Complete the multi processing of getting location accesses for all satellite tasks
    mpctx = mp.get_context('fork')
    with mpctx.Pool(mp.cpu_count()) as pool:
        all_results = []
        for sat_tasks in grouped_tasks:
            sat_results = pool.starmap(ba.find_location_accesses, sat_tasks)
            all_results.append(sat_results)

    # For all contacts on satellites, condense into one large list
    # Also useful for plotting
    for id_gs,sat_result in enumerate(all_results):
        all_contacts_sat = []
        for id_sat,contacts in enumerate(sat_result):
            
            # Plotting per satellite and gs contact windows
            if plot:
                gaps = gap_times(contacts, epc_start.to_datetime(),epc_end.to_datetime())
                ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (id_sat+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:blue',label = "contacts")
                ax.broken_barh([(gaps[i][0],gaps[i][1]-gaps[i][0]) for i in range(len(gaps))], (id_sat+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:green',alpha = 0.5,label = "gaps")

            # Condense contacts
            for contact in contacts:
                all_contacts_sat.append(contact)
                if plot:
                    ax.broken_barh([(contacts[i].t_start,contacts[i].t_end-contacts[i].t_start) for i in range(len(contacts))], (-1+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:purple')

        # Condensing gap times
        all_gaps_sat, all_gaps_sat_secs = gap_times_condense(all_contacts_sat,epc_start.to_datetime(),epc_end.to_datetime())

        # Plotting condensed gap times
        if plot:
            ax.broken_barh([(all_gaps_sat[i][0],all_gaps_sat[i][1]-all_gaps_sat[i][0]) for i in range(len(all_gaps_sat))], (-1+0.05 + len_gs*(id_gs-1), 0.85),facecolors='tab:green', alpha = 0.5)

        # Saving all satellite outputs into full constellation outputs
        all_contacts.append(all_contacts_sat)
        all_gaps.append(all_gaps_sat)
        all_gaps_secs.append(all_gaps_sat_secs)

    # Plotting
    if plot:
        # ax.broken_barh([(all_gap_times[i][0],all_gap_times[i][1]-all_gap_times[i][0]) for i in range(len(all_gap_times))], (-1.9, 0.85),facecolors='tab:green',label = "gaptimes")
        plt.ylabel("Different Satellite groups, With Contact / Gap Times per Ground station")
        plt.xlabel("Time period over a day")
        # plt.legend()
        plt.savefig(title)
        
    return all_contacts, all_gaps, all_gaps_secs



# TODO: Need to edit, can't use this specific format
def load_earth_data(filename, download=False):
    # Here we can download the latest Earth orientation data and load it.

    if not os.path.exists(filename):
        if download:
            # Uncomment this line ONCE the data has been downloaded. Recomment it once it has been downloaded.
            print("Downloading latest earth orientation data...")
            if not os.path.exists("data"):
                os.makedirs("data")
            # bh.utils.download_iers_bulletin_ab("./data")
            print("Download complete")

    # Load the latest Earth Orientation Data
    print("Loading the latest Earth Orientation Data")
    bh.initialize_eop()