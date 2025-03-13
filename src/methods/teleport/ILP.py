# TODO
# Standard imports
import sys
import os
from itertools import groupby

# Add the path to the folder containing the module
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Required imports
from common.sat_gen import make_tle
from common.station_gen import teleport_json
from common.utils import load_earth_data, gap_times_condense

import brahe as bh
import brahe.data_models as bdm
import brahe.access.access as ba
import numpy as np
import matplotlib.pyplot as plt
import json

# Pyomo imports
import pyomo.environ as pyo
import pyomo.kernel as pk



def data_downlink_ilp(ground_stations, sat1, epc0, epc1, max_stations):
    model = pk.block()
    # Load original station data from JSON
    with open('data/teleport_locations.json', 'r') as f:
        stations_data = json.load(f)

    # Get contacts for each station
    station_contacts = {}
    data_rate = 10  # Mbps (example value)
    for i, station in enumerate(ground_stations):
        contacts = ba.find_location_accesses(sat1, station, epc0, epc1)
        # Calculate data volume for each contact
        data_volumes = [(contact.t_end - contact.t_start).total_seconds() * data_rate for contact in contacts]
        station_contacts[i] = {
            'contacts': contacts,
            'station': station,
            'name': stations_data[i]['name'],  # Get name from original JSON
            'data_volumes': data_volumes,
            'total_data': sum(data_volumes)
        }

    # Create binary variables for station selection
    model.x = pk.variable_dict()
    for i in range(len(ground_stations)):
        model.x[i] = pk.variable(domain=pk.Binary)

    # Create constraints list
    model.constraints = pk.constraint_list()

    # Constraint for exact number of stations
    MAX_STATIONS = max_stations
    model.constraints.append(pk.constraint(sum(model.x.values()) == MAX_STATIONS))

    # Create variable for total data volume
    model.total_data = pk.variable(domain=pk.NonNegativeReals)

    # Calculate total data volume expression
    total_data_expr = sum(station_contacts[i]['total_data'] * model.x[i] for i in range(len(ground_stations)))

    # Set total data constraint
    model.constraints.append(pk.constraint(model.total_data == total_data_expr))

    # Create and attach objective to model - maximize total data
    model.obj = pk.objective(model.total_data, sense=pk.maximize)

    # Create solver using SolverFactory
    from pyomo.opt import SolverFactory
    solver = SolverFactory('gurobi')

    # Solve
    solver.solve(model)

    # Print results
    selected_stations = []
    total_data = 0
    print("\nSelected stations and their data volumes:")
    for i in model.x:
        if model.x[i].value > 0.5:
            selected_stations.append(i)
            station_data = station_contacts[i]['total_data']
            total_data += station_data
            
    return selected_stations, total_data, station_contacts


def gap_time_ilp(ground_stations, sat1, epc0, epc1, max_stations):
    model = pk.block()
    MAX_STATIONS = max_stations
    # Load original station data from JSON
    with open('data/teleport_locations.json', 'r') as f:
        stations_data = json.load(f)

    # Get contacts for each station
    station_contacts = {}
    for i, station in enumerate(ground_stations):
        contacts = ba.find_location_accesses(sat1, station, epc0, epc1)
        station_contacts[i] = {
            'contacts': contacts,
            'station': station,
            'name': stations_data[i]['name']  # Get name from original JSON
        }
    # Create binary variables for station selection using kernel interface
    model.x = pk.variable_dict()
    for i in range(len(ground_stations)):
        model.x[i] = pk.variable(domain=pk.Binary)

    # Create constraints list
    model.constraints = pk.constraint_list()

    # Constraint for exact number of stations
    model.constraints.append(pk.constraint(sum(model.x.values()) == MAX_STATIONS))

    # Create variable for total gap time
    model.total_gap = pk.variable(domain=pk.NonNegativeReals)

    # Create variables for all possible combinations of stations
    from itertools import combinations
    station_combinations = list(combinations(range(len(ground_stations)), MAX_STATIONS))
    # Binary variable for each combination
    model.combo_vars = pk.variable_dict()
    for combo in station_combinations:
        model.combo_vars[combo] = pk.variable(domain=pk.Binary)

    # Only one combination can be selected
    model.constraints.append(pk.constraint(sum(model.combo_vars.values()) == 1))

    # Link station selection variables with combination variables
    for combo in station_combinations:
        # If combo is selected, corresponding stations must be selected
        for i in combo:
            model.constraints.append(pk.constraint(model.x[i] >= model.combo_vars[combo]))
        # If all stations in combo are selected, combo must be selected
        model.constraints.append(
            pk.constraint(sum(model.x[i] for i in combo) - MAX_STATIONS * model.combo_vars[combo] <= MAX_STATIONS - 1)
        )

    # Calculate total gap time for each combination using gap_times_condense
    total_gap_expr = 0
    for combo in station_combinations:
        # Collect all contacts for this combination
        combo_contacts = []
        for station_id in combo:
            combo_contacts.extend(station_contacts[station_id]['contacts'])
        
        # Use gap_times_condense to calculate gaps
        _, gaps_seconds = gap_times_condense(combo_contacts, epc0.to_datetime(), epc1.to_datetime())
        combo_gap = sum(gaps_seconds)
        
        # Add this combination's contribution to total gap
        total_gap_expr += combo_gap * model.combo_vars[combo]

    # Set total gap constraint
    model.constraints.append(pk.constraint(model.total_gap == total_gap_expr))

    # Create and attach objective to model
    model.obj = pk.objective(model.total_gap, sense=pk.minimize)

    # Create solver using SolverFactory
    from pyomo.opt import SolverFactory
    solver = SolverFactory('gurobi')

    # Solve
    solver.solve(model)

    # Print results
    selected_stations = []
    for i in model.x:
        if model.x[i].value > 0.5:
            selected_stations.append(i)

    # Calculate and print actual gaps using gap_times_condense
    selected_contacts = []
    for station_id in selected_stations:
        selected_contacts.extend(station_contacts[station_id]['contacts'])


    all_gap_times, gaps_seconds = gap_times_condense(selected_contacts, epc0.to_datetime(), epc1.to_datetime())
    print("\nGaps between merged contacts:")
    for i, gap_time in enumerate(gaps_seconds):
        print(f"Gap {i+1}: {gap_time:.2f} seconds")

    print(f"\nTotal gap time: {sum(gaps_seconds):.2f} seconds")
    print([station_contacts[i]['name'] for i in selected_stations])
    return selected_stations, station_contacts, all_gap_times

def max_contact_ilp(ground_stations, sat1, epc0, epc1, max_stations):
    model = pk.block()
    # Load original station data from JSON
    with open('data/teleport_locations.json', 'r') as f:
        stations_data = json.load(f)

    # Get contacts for each station
    station_contacts = {}
    for i, station in enumerate(ground_stations):
        contacts = ba.find_location_accesses(sat1, station, epc0, epc1)
        print(f"Station {i} has {len(contacts)} contacts")
        # Calculate data volume for each contact
        station_contacts[i] = {
            'contacts': contacts,
            'station': station,
            'name': stations_data[i]['name']  # Get name from original JSON
        }

    # Create binary variables for station selection
    model.x = pk.variable_dict()
    for i in range(len(ground_stations)):
        model.x[i] = pk.variable(domain=pk.Binary)

    # Create constraints list
    model.constraints = pk.constraint_list()

    # Constraint for exact number of stations
    MAX_STATIONS = max_stations
    model.constraints.append(pk.constraint(sum(model.x.values()) == MAX_STATIONS))

    # Create variable for total data volume
    model.total_contacts = pk.variable(domain=pk.NonNegativeReals)

    # Calculate total data volume expression
    total_contacts_expr = sum(len(station_contacts[i]['contacts']) * model.x[i] for i in range(len(ground_stations)))

    # Set total data constraint
    model.constraints.append(pk.constraint(model.total_contacts == total_contacts_expr))

    # Create and attach objective to model - maximize total data
    model.obj = pk.objective(model.total_contacts, sense=pk.maximize)

    # Create solver using SolverFactory
    from pyomo.opt import SolverFactory
    solver = SolverFactory('gurobi')

    # Solve
    solver.solve(model)

    # Print results
    selected_stations = []
    total_contacts = 0
    print("\nSelected stations and their number of contacts:")
    for i in model.x:
        if model.x[i].value > 0.5:
            selected_stations.append(i)
            station_data = len(station_contacts[i]['contacts'])
            total_contacts += station_data
            print(f"Station {i} ({station_contacts[i]['name']}): {station_data:.2f} Contacts")
            
    print(f"\nTotal number of contacts: {total_contacts:.2f} Contacts")
    print([station_contacts[i]['name'] for i in selected_stations])
    return selected_stations, station_contacts, total_contacts
