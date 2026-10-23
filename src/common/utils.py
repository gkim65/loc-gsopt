import os.path
from common.paths import repo_path

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
    """
        xyz_to_latlon(xyz) -> numpy.ndarray

    Convert a Cartesian direction to spherical (not geodetic) latitude and longitude.

      - `xyz` — 3-vector in any length units; it is normalized first

    Returns `[lat, lon]` (deg).

    NOTE: the order is latitude first, the reverse of the `[lon, lat]` pairs used elsewhere.
    """
    xyz = xyz / np.linalg.norm(xyz)
    x, y, z = xyz
    lat_rad = np.arcsin(z)  # sin(lat) = z
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.array([lat_deg, lon_deg])


def latlon_to_xyz(lat_deg, lon_deg):
    """
        latlon_to_xyz(lat_deg, lon_deg) -> numpy.ndarray

    Convert spherical latitude and longitude to a unit vector.

      - `lat_deg` — latitude (deg)
      - `lon_deg` — longitude (deg)

    Returns the unit 3-vector `[x, y, z]`.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.array([x, y, z])

def contactExclusion(contacts, cfg):
    """
        contactExclusion(contacts, cfg) -> (list, list)

    Select the subset of contacts that can actually be used, by solving a binary program:
    no two selected contacts may overlap on the same satellite or the same ground station, and
    contacts of 180 s or less are never selected.

      - `contacts` — access windows from `compute_contact_times`
      - `cfg` — the Hydra config; `cfg.problem.objective == "data_downlink"` maximizes total
        selected contact time

    Returns the selected contacts and their durations (s).

    NOTE: solves with `cfg.problem.solver` (default "cbc"), whose executable must be on PATH.
    """
    if not contacts:
        return [], []

    model = ConcreteModel()

    # Contacts are keyed by their index in `contacts`
    contacts_order = {id(contact): i for i, contact in enumerate(contacts)}

    # Decision variables: x[c] = 1 if contact c is selected, 0 otherwise
    model.x = Var(range(len(contacts)), within=Binary)

    if cfg.problem.objective == "maximize_num_contacts":
        model.obj = Objective(
            expr=sum(model.x[i] for i in range(len(contacts))),
            sense=maximize
        )

    if cfg.problem.objective == "data_downlink":
        model.obj = Objective(
            expr=sum(model.x[i] * contact.duration
                    for i, contact in enumerate(contacts)),
            sense=maximize
        )

    # ── Build overlap pairs ───────────────────────────────────────────
    overlap_pairs = []

    # Satellite exclusion
    contacts_sorted_by_satellite = sorted(
        contacts, key=lambda cn: cn.satellite_id
    )
    for sat_id, satellite_contacts in groupby(
        contacts_sorted_by_satellite, lambda cn: cn.satellite_id
    ):
        satellite_contacts = sorted(
            list(satellite_contacts), key=lambda cn: cn.start
        )
        
        for x, y in combinations(satellite_contacts, 2):
            if x.start <= y.end and y.start <= x.end:
                overlap_pairs.append(
                    (contacts_order[id(x)], contacts_order[id(y)])
                )

    # Ground station exclusion
    contacts_sorted_by_gs = sorted(
        contacts,
        key=lambda cn: cn.location_id if cn.location_id is not None else ""
    )
    for gs_id, gs_contacts in groupby(
        contacts_sorted_by_gs, lambda cn: cn.location_id
    ):
        gs_contacts = sorted(list(gs_contacts), key=lambda cn: cn.start)
        for x, y in combinations(gs_contacts, 2):
            if x.start <= y.end and y.start <= x.end:
                overlap_pairs.append(
                    (contacts_order[id(x)], contacts_order[id(y)])
                )

    # Deduplicate pairs
    overlap_pairs_set = list(set(overlap_pairs))

    # ── Add constraints explicitly ────────────────────────────────────
    for k, (i, j) in enumerate(overlap_pairs_set):
        model.add_component(
            f'c_{k}',
            Constraint(expr=model.x[i] + model.x[j] <= 1)
        )

    # Minimum duration constraint
    for i, contact in enumerate(contacts):
        if contact.duration <= 180:
            model.add_component(
                f'md_{i}',
                Constraint(expr=model.x[i] == 0)
            )

    # ── Debug prints ──────────────────────────────────────────────────
    print(f"Total contacts: {len(contacts)}")
    print(f"Total overlap pairs: {len(overlap_pairs)}")
    print(f"Unique overlap pairs: {len(overlap_pairs_set)}")

    # ── Solve ─────────────────────────────────────────────────────────
    solver = SolverFactory(cfg.problem.get("solver", "cbc"))
    solver_result = solver.solve(model)
    print(solver_result)
    print(f"Selected contacts: {sum(1 for i in range(len(contacts)) if value(model.x[i]) > 0.75)}")
    print(f"Total contacts: {len(contacts)}")

    # Extract selected contacts
    selected_contacts = [
        contact for i, contact in enumerate(contacts)
        if value(model.x[i]) > 0.75
    ]
    contact_times_seconds = [contact.duration for contact in selected_contacts]

    # ── Sanity check ──────────────────────────────────────────────────
    violations = 0
    sat_selected = {}
    for contact in selected_contacts:
        sid = contact.satellite_id
        if sid not in sat_selected:
            sat_selected[sid] = []
        for prev in sat_selected[sid]:
            if prev.start <= contact.end and contact.start <= prev.end:
                violations += 1
        sat_selected[sid].append(contact)
    print(f"Constraint violations: {violations}")

    return selected_contacts, contact_times_seconds

def compute_contact_times(satellites, ground_stations, epc_start, epc_end, min_elevation_deg=10.0):
    """
        compute_contact_times(satellites, ground_stations, epc_start, epc_end, min_elevation_deg=10.0) -> (list, list)

    Find every access window between the satellites and the ground stations.

      - `satellites` — `bh.SGPPropagator` objects
      - `ground_stations` — `bh.PointLocation` objects
      - `epc_start`, `epc_end` — search window (`bh.Epoch`)
      - `min_elevation_deg` — minimum elevation of the satellite seen from the station (deg)

    Returns the access windows and their durations (s).

    NOTE: the satellites must already be propagated to `epc_end` (`bh.par_propagate_to`).
    """
    constraint = bh.ElevationConstraint(min_elevation_deg)

    windows = bh.location_accesses(
        ground_stations,
        satellites,
        epc_start,
        epc_end,
        constraint
    )

    all_contacts = []
    contact_times_seconds = []

    for c in windows:
        all_contacts.append(c)
        contact_times_seconds.append(c.t_end - c.t_start)

    return all_contacts, contact_times_seconds

def compute_earth_interior_angle(ele=0.0, alt=525):
    """
        compute_earth_interior_angle(ele=0.0, alt=525) -> float

    Earth central angle between the sub-satellite point and an observer who sees the satellite
    at elevation `ele`. Multiplied by the Earth radius it gives the radius of the coverage
    circle drawn on the station maps.

      - `ele` — elevation of the satellite seen by the observer (deg)
      - `alt` — satellite altitude (km)

    Returns the central angle (rad).
    """
    ele = ele * math.pi / 180.0

    rho = math.asin(bh.R_EARTH/(bh.R_EARTH + alt * 1e3))

    eta = math.asin(math.cos(ele)*math.sin(rho))
    lam = math.pi/2.0 - eta - ele

    return lam



def gap_times(contacts, epc_start, epc_end):
    """
        gap_times(contacts, epc_start, epc_end) -> list

    Gaps between consecutive contacts of one satellite-station pair, plus the gaps before the
    first and after the last contact.

      - `contacts` — contacts sorted by start time
      - `epc_start`, `epc_end` — simulation window, same time type as the contact times

    Returns `(gap_start, gap_end)` pairs.

    NOTE: used only by the gap-time objective, which is not currently supported.
    """
    gap_time_list = [(contacts[i].t_end,contacts[i+1].t_start) for i in range(len(contacts)-1)]
    
    if len(contacts) != 0:
        if epc_start < contacts[0].t_start:
            gap_time_list.append((epc_start,contacts[0].t_start))
        if epc_end > contacts[-1].t_end:
            gap_time_list.append((contacts[-1].t_end,epc_end))
    
    return gap_time_list


def gap_times_condense(contacts_all, epc_start, epc_end):
    """
        gap_times_condense(contacts_all, epc_start, epc_end) -> (list, list)

    Gaps in coverage for one satellite across all its stations, merging overlapping contacts.

      - `contacts_all` — every contact of one satellite, in any order
      - `epc_start`, `epc_end` — simulation window (`datetime`)

    Returns the `(gap_start, gap_end)` pairs and the gap durations (s).

    NOTE: expects contacts whose `t_start`/`t_end` subtract to a `timedelta`, as in the pre-1.0
    brahe API; used only by the gap-time objective and the MILP path, neither currently supported.
    """
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


def compute_gaps_per_sat(satellites,ground_stations,epc_start,epc_end, plot,title="gap_times_chart.png"):
    """
        compute_gaps_per_sat(satellites, ground_stations, epc_start, epc_end, plot, title="gap_times_chart.png") -> (list, list, list)

    Per-satellite contacts and coverage gaps over all ground stations, optionally plotted.

      - `satellites`, `ground_stations` — spacecraft and stations
      - `epc_start`, `epc_end` — simulation window (`bh.Epoch`)
      - `plot` — save a contact/gap chart to `title` when True
      - `title` — output image path

    Returns per-satellite lists of contacts, gap intervals and gap durations (s).

    NOTE: calls the pre-1.0 brahe `find_location_accesses` (`ba` is not imported), so it raises
    `NameError`; the gap-time objective that uses it is not currently supported.
    """

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



def load_earth_data(filename, download=False):
    """
        load_earth_data(filename, download=False) -> None

    Initialize brahe's default Earth orientation data.

      - `filename` — EOP file path; only checked for existence
      - `download` — print download messages when `filename` is missing

    NOTE: nothing is downloaded and `filename` is not loaded; `bh.initialize_eop()` uses brahe's
    built-in data. `main.setup_scenario` loads `data/iau2000A_finals_ab.txt` explicitly instead.
    """
    if not os.path.exists(filename):
        if download:
            print("Downloading latest earth orientation data...")
            if not os.path.exists(repo_path("data")):
                os.makedirs(repo_path("data"))
            # bh.utils.download_iers_bulletin_ab(repo_path("data"))
            print("Download complete")

    # Load the latest Earth Orientation Data
    print("Loading the latest Earth Orientation Data")
    bh.initialize_eop()