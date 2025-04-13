# TODO
# Standard imports
import sys
import os
from itertools import groupby, combinations

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

class ILP_Model:
    def __init__(self, ground_stations, satellites, epc0, epc1, max_stations, data_rate=10):
        self.ground_stations = ground_stations
        self.satellites = satellites
        self.epc0 = epc0
        self.epc1 = epc1
        self.max_stations = max_stations
        self.data_rate = data_rate
        #Initialize model
        self.model = pk.block()
        with open('data/teleport_locations.json', 'r') as f:
            self.stations_data = json.load(f)

        self.station_contacts = {}
        self.model.x = pk.variable_dict()
                # Create binary variables for station selection
        for i in range(len(self.ground_stations)):
            self.model.x[i] = pk.variable(domain=pk.Binary)

        # Create constraints list
        self.model.constraints = pk.constraint_list()

        # Constraint for exact number of stations
        self.model.constraints.append(pk.constraint(sum(self.model.x.values()) == self.max_stations))

        

    def data_downlink_ilp(self):
        # Initialize empty list to store all contacts
        all_contacts = []

        for i, station in enumerate(self.ground_stations):
            #Calculate data volumes for each contact
            self.station_contacts[i] = {
                'contacts': [],
                'station': station,
                'name': self.stations_data[i]['name'],  # Get name from original JSON
                'data_volumes': [],
                'total_data': 0
            }
            for sat in self.satellites:
                contacts = ba.find_location_accesses(sat, station, self.epc0, self.epc1)                    
                data_volumes = [(contact.t_end - contact.t_start).total_seconds() * self.data_rate for contact in contacts]
                self.station_contacts[i]['contacts'].extend(contacts)
                self.station_contacts[i]['data_volumes'].extend(data_volumes)
                all_contacts.extend(contacts)  # Add contacts to master list
            self.station_contacts[i]['total_data'] += sum(self.station_contacts[i]['data_volumes'])
            
        # Create binary variables for each contact
        self.model.contact_vars = pk.variable_dict()
        contacts_order = {
            str(contact.id): i 
            for i, contact in enumerate(all_contacts)
        }
        for contact in all_contacts:
            self.model.contact_vars[contacts_order[str(contact.id)]] = pk.variable(domain=pk.Binary)
        
        # Add constraints for overlapping contacts at each station
        contacts_sorted_by_station = sorted(all_contacts, key=lambda cn: cn.station_id)
        for station_id, station_contacts in groupby(contacts_sorted_by_station, lambda cn: cn.station_id):
            station_contacts = sorted(list(station_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(station_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for overlapping contacts at each satellite
        contacts_sorted_by_satellite = sorted(all_contacts, key=lambda cn: cn.spacecraft_id)
        for sat_id, satellite_contacts in groupby(contacts_sorted_by_satellite, lambda cn: cn.spacecraft_id):
            satellite_contacts = sorted(list(satellite_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(satellite_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for minimum contact duration
        for contact in all_contacts:
            if contact.t_duration < 180:  # Contacts must be longer than 3 minutes
                self.model.constraints.append(
                    pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] == 0)
                )

        # Create variable for total data volume
        self.model.total_data = pk.variable(domain=pk.NonNegativeReals)

        # Link contact variables with station selection
        
        for contact in all_contacts:
            station_idx = next(i for i, station in enumerate(self.ground_stations) 
                             if station.id == contact.station_id)
            self.model.constraints.append(
                pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] <= self.model.x[station_idx])
            )
        
        
        

        # Calculate total data volume expression
        total_data_expr = sum(
            self.model.contact_vars[contacts_order[str(contact.id)]] * 
            contact.t_duration * self.data_rate 
            for contact in all_contacts
        )
        
        # Set total data constraint
        self.model.constraints.append(pk.constraint(self.model.total_data == total_data_expr))

        # Create and attach objective to model - maximize total data
        self.model.obj = pk.objective(self.model.total_data, sense=pk.maximize)

        self.solve()
        
        #Handle output
        selected_stations = []
        total_data = 0
        print("\nSelected stations and their data volumes:")
        selected_contacts = []
        for i in self.model.contact_vars:
            if self.model.contact_vars[i].value > 0.5:
                selected_contacts.append(all_contacts[i])

        for i in self.model.x:
            if self.model.x[i].value > 0.5:
                selected_stations.append(i)
                station_data = self.station_contacts[i]['total_data']
                total_data += station_data
                


           
        return selected_stations, self.station_contacts, self.model.total_data.value


    def solve(self):
        # Create solver using SolverFactory
        from pyomo.opt import SolverFactory
        solver = SolverFactory('gurobi')

        # Solve
        solver.solve(self.model)

        


    def gap_time_ilp(self):
        # Initialize empty list to store all contacts
        all_contacts = []

        for i, station in enumerate(self.ground_stations):
            self.station_contacts[i] = {
                'contacts': [],
                'station': station,
                'name': self.stations_data[i]['name']  # Get name from original JSON
            }
            for sat in self.satellites:
                contacts = ba.find_location_accesses(sat, station, self.epc0, self.epc1)
                self.station_contacts[i]['contacts'].extend(contacts)
                all_contacts.extend(contacts)

        # Create binary variables for each contact
        self.model.contact_vars = pk.variable_dict()
        contacts_order = {
            str(contact.id): i 
            for i, contact in enumerate(all_contacts)
        }
        for contact in all_contacts:
            self.model.contact_vars[contacts_order[str(contact.id)]] = pk.variable(domain=pk.Binary)
        
        # Add constraints for overlapping contacts at each station
        contacts_sorted_by_station = sorted(all_contacts, key=lambda cn: cn.station_id)
        for station_id, station_contacts in groupby(contacts_sorted_by_station, lambda cn: cn.station_id):
            station_contacts = sorted(list(station_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(station_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for overlapping contacts at each satellite
        contacts_sorted_by_satellite = sorted(all_contacts, key=lambda cn: cn.spacecraft_id)
        for sat_id, satellite_contacts in groupby(contacts_sorted_by_satellite, lambda cn: cn.spacecraft_id):
            satellite_contacts = sorted(list(satellite_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(satellite_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for minimum contact duration
        for contact in all_contacts:
            if contact.t_duration < 180:  # Contacts must be longer than 3 minutes
                self.model.constraints.append(
                    pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] == 0)
                )

        # Link contact variables with station selection
        for contact in all_contacts:
            station_idx = next(i for i, station in enumerate(self.ground_stations) 
                             if station.id == contact.station_id)
            self.model.constraints.append(
                pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] <= self.model.x[station_idx])
            )

        # Create variable for total gap time
        self.model.total_gap = pk.variable(domain=pk.NonNegativeReals)

        # First solve the model to select stations
        self.solve()

        # Now get selected contacts after solving
        selected_contacts = []
        for i in self.model.contact_vars:
            if self.model.contact_vars[i].value > 0.5:
                selected_contacts.append(all_contacts[i])

        # Calculate gaps based on selected contacts
        if len(selected_contacts) > 0:
            all_gap_times, gaps_seconds = gap_times_condense(selected_contacts, self.epc0.to_datetime(), self.epc1.to_datetime())
            total_gap_expr = sum(gaps_seconds)
        else:
            # If no contacts selected, set a large gap penalty
            total_gap_expr = (self.epc1.to_datetime() - self.epc0.to_datetime()).total_seconds()
            all_gap_times = []
            gaps_seconds = [total_gap_expr]

        # Set total gap constraint and objective
        self.model.constraints.append(pk.constraint(self.model.total_gap == total_gap_expr))
        self.model.obj = pk.objective(self.model.total_gap, sense=pk.minimize)

        # Solve again with the gap objective
        self.solve()

        # Handle output
        selected_stations = []
        selected_contacts = []
        for i in self.model.contact_vars:
            if self.model.contact_vars[i].value > 0.5:
                selected_contacts.append(all_contacts[i])

        for i in self.model.x:
            if self.model.x[i].value > 0.5:
                selected_stations.append(i)

        # Recalculate gaps for final output
        if len(selected_contacts) > 0:
            all_gap_times, gaps_seconds = gap_times_condense(selected_contacts, self.epc0.to_datetime(), self.epc1.to_datetime())
        else:
            all_gap_times = []
            gaps_seconds = [(self.epc1.to_datetime() - self.epc0.to_datetime()).total_seconds()]

        print("\nGaps between merged contacts:")
        for i, gap_time in enumerate(gaps_seconds):
            print(f"Gap {i+1}: {gap_time:.2f} seconds")

        print(f"\nTotal gap time: {sum(gaps_seconds):.2f} seconds")
        print([self.station_contacts[i]['name'] for i in selected_stations])
        
        return selected_stations, self.station_contacts, all_gap_times

    def max_contact_ilp(self):
        # Initialize empty list to store all contacts
        all_contacts = []

        for i, station in enumerate(self.ground_stations):
            self.station_contacts[i] = {
                'contacts': [],
                'station': station,
                'name': self.stations_data[i]['name']  # Get name from original JSON
            }
            for sat in self.satellites:
                contacts = ba.find_location_accesses(sat, station, self.epc0, self.epc1)
                self.station_contacts[i]['contacts'].extend(contacts)
                all_contacts.extend(contacts)

        # Create binary variables for each contact
        self.model.contact_vars = pk.variable_dict()
        contacts_order = {
            str(contact.id): i 
            for i, contact in enumerate(all_contacts)
        }
        for contact in all_contacts:
            self.model.contact_vars[contacts_order[str(contact.id)]] = pk.variable(domain=pk.Binary)
        
        # Add constraints for overlapping contacts at each station
        contacts_sorted_by_station = sorted(all_contacts, key=lambda cn: cn.station_id)
        for station_id, station_contacts in groupby(contacts_sorted_by_station, lambda cn: cn.station_id):
            station_contacts = sorted(list(station_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(station_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for overlapping contacts at each satellite
        contacts_sorted_by_satellite = sorted(all_contacts, key=lambda cn: cn.spacecraft_id)
        for sat_id, satellite_contacts in groupby(contacts_sorted_by_satellite, lambda cn: cn.spacecraft_id):
            satellite_contacts = sorted(list(satellite_contacts), key=lambda cn: cn.t_start)
            
            for x, y in combinations(satellite_contacts, 2):
                if x.t_start <= y.t_end and y.t_start <= x.t_end:
                    expr = self.model.contact_vars[contacts_order[str(x.id)]] + self.model.contact_vars[contacts_order[str(y.id)]]
                    self.model.constraints.append(pk.constraint(expr <= 1))
        
        # Add constraints for minimum contact duration
        for contact in all_contacts:
            if contact.t_duration < 180:  # Contacts must be longer than 3 minutes
                self.model.constraints.append(
                    pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] == 0)
                )

        # Link contact variables with station selection
        for contact in all_contacts:
            station_idx = next(i for i, station in enumerate(self.ground_stations) 
                             if station.id == contact.station_id)
            self.model.constraints.append(
                pk.constraint(self.model.contact_vars[contacts_order[str(contact.id)]] <= self.model.x[station_idx])
            )

        # Create variable for total contacts
        self.model.total_contacts = pk.variable(domain=pk.NonNegativeReals)

        # Calculate total contacts expression
        total_contacts_expr = sum(self.model.contact_vars.values())

        # Set total contacts constraint
        self.model.constraints.append(pk.constraint(self.model.total_contacts == total_contacts_expr))

        # Create and attach objective to model
        self.model.obj = pk.objective(self.model.total_contacts, sense=pk.maximize)

        self.solve()

        # Handle output
        selected_stations = []
        selected_contacts = []
        total_contacts = 0
        
        for i in self.model.contact_vars:
            if self.model.contact_vars[i].value > 0.5:
                selected_contacts.append(all_contacts[i])
                total_contacts += 1

        print("\nSelected stations and their number of contacts:")
        for i in self.model.x:
            if self.model.x[i].value > 0.5:
                selected_stations.append(i)
                print(f"Station {i} ({self.station_contacts[i]['name']})")
                
        print(f"\nTotal number of contacts: {total_contacts}")
        print([self.station_contacts[i]['name'] for i in selected_stations])
        
        return selected_stations, self.station_contacts, total_contacts
