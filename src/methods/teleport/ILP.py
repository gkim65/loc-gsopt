"""
MILP baseline: choose `max_stations` sites from a fixed candidate list to maximize data downlinked.
"""

from itertools import groupby, combinations

from common.paths import repo_path
from common.utils import compute_contact_times, gap_times_condense

import brahe as bh
import json

import pyomo.kernel as pk

class ILP_Model:
    """
        ILP_Model(ground_stations, satellites, epc0, epc1, max_stations, data_rate=10, solver="cbc")

    Fixed-site baseline: a binary program that selects exactly `max_stations` of the candidate
    sites, built with `pyomo.kernel`.

      - `ground_stations` — candidate sites from `teleport_json`, ids equal to their list index
      - `satellites` — propagated `bh.SGPPropagator` objects
      - `epc0`, `epc1` — simulation window (`bh.Epoch`)
      - `max_stations` — number of sites to select
      - `data_rate` — downlink rate (bps)
      - `solver` — Pyomo solver name, "cbc" (free) or "gurobi" (needs a license)

    NOTE: site names come from `data/most_diverse_100_ground_stations.json` by index, so
    `ground_stations` must be a prefix of that file.
    """
    def __init__(self, ground_stations, satellites, epc0, epc1, max_stations, data_rate=10, solver="cbc"):
        self.ground_stations = ground_stations
        self.satellites = satellites
        self.epc0 = epc0
        self.epc1 = epc1
        self.max_stations = max_stations
        self.data_rate = data_rate
        self.solver = solver
        #Initialize model
        self.model = pk.block()
        with open(repo_path('data/most_diverse_100_ground_stations.json'), 'r') as f:
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
        """
            data_downlink_ilp() -> (list, dict, float)

        Select exactly `max_stations` candidate sites and the usable contacts at them to
        maximize data downlinked. Selected contacts may not overlap on the same station or the
        same satellite, contacts shorter than 180 s are excluded, and a contact can only be
        selected at a selected station.

        Returns the selected station indices, per-station contact records, and the total data
        downlinked (bits, selected contact seconds times `data_rate` in bps).
        """
        all_contacts, _ = compute_contact_times(self.satellites, self.ground_stations, self.epc0, self.epc1)

        for i, station in enumerate(self.ground_stations):
            self.station_contacts[i] = {
                'contacts': [],
                'station': station,
                'name': self.stations_data[i]['name'],  # Get name from original JSON
                'data_volumes': [],
                'total_data': 0
            }
        # Stations carry their list index as id (see teleport_json)
        for contact in all_contacts:
            record = self.station_contacts[contact.location_id]
            record['contacts'].append(contact)
            record['data_volumes'].append(contact.duration * self.data_rate)
        for record in self.station_contacts.values():
            record['total_data'] = sum(record['data_volumes'])

        # Create binary variables for each contact, keyed by its index in all_contacts
        self.model.contact_vars = pk.variable_dict()
        for k in range(len(all_contacts)):
            self.model.contact_vars[k] = pk.variable(domain=pk.Binary)
        indexed = list(enumerate(all_contacts))

        # No two selected contacts may overlap at the same station, or on the same satellite
        for key in (lambda kc: kc[1].location_id, lambda kc: kc[1].satellite_id):
            for _, group in groupby(sorted(indexed, key=key), key):
                group = sorted(group, key=lambda kc: kc[1].t_start)
                for (kx, x), (ky, y) in combinations(group, 2):
                    if x.t_start <= y.t_end and y.t_start <= x.t_end:
                        self.model.constraints.append(
                            pk.constraint(self.model.contact_vars[kx] + self.model.contact_vars[ky] <= 1))

        # Add constraints for minimum contact duration
        for k, contact in indexed:
            if contact.duration < 180:  # Contacts must be longer than 3 minutes
                self.model.constraints.append(pk.constraint(self.model.contact_vars[k] == 0))

        # Link contact variables with station selection
        for k, contact in indexed:
            self.model.constraints.append(
                pk.constraint(self.model.contact_vars[k] <= self.model.x[contact.location_id]))

        # Total data volume, maximized
        self.model.total_data = pk.variable(domain=pk.NonNegativeReals)
        total_data_expr = sum(self.model.contact_vars[k] * contact.duration * self.data_rate
                              for k, contact in indexed)
        self.model.constraints.append(pk.constraint(self.model.total_data == total_data_expr))
        self.model.obj = pk.objective(self.model.total_data, sense=pk.maximize)

        self.solve()

        selected_stations = [i for i in self.model.x if self.model.x[i].value > 0.5]
        print("contacts", sum(1 for k in self.model.contact_vars if self.model.contact_vars[k].value > 0.5))
        return selected_stations, self.station_contacts, self.model.total_data.value


    def solve(self):
        """solve() -> None: solve the built model in place with `self.solver`."""
        # Create solver using SolverFactory
        from pyomo.opt import SolverFactory
        solver = SolverFactory(self.solver)

        # Solve
        solver.solve(self.model)

        
    def gap_time_ilp(self):
         """
             gap_time_ilp() -> (list, dict, list)

         Select `max_stations` sites to minimize total coverage gap time.

         NOTE: still written against the pre-1.0 brahe access API (`ba` is not imported); the
         gap-time objective is not currently supported.
         """
 
         # Get contacts for each station
 
         for i, station in enumerate(self.ground_stations):
             self.station_contacts[i] = {
                 'contacts': [],
                 'station': station,
                 'name': self.stations_data[i]['name']  # Get name from original JSON
             }
 
             for sat in self.satellites:
                 contacts = ba.find_location_accesses(sat, station, self.epc0, self.epc1)
                 self.station_contacts[i]['contacts'].extend(contacts)
 
 
 
         # Create variable for total gap time
         self.model.total_gap = pk.variable(domain=pk.NonNegativeReals)
 
         # Create variables for all possible combinations of stations
         station_combinations = list(combinations(range(len(self.ground_stations)), self.max_stations))
         # Binary variable for each combination
         self.model.combo_vars = pk.variable_dict()
         for combo in station_combinations:
             self.model.combo_vars[combo] = pk.variable(domain=pk.Binary)
 
         # Only one combination can be selected
         self.model.constraints.append(pk.constraint(sum(self.model.combo_vars.values()) == 1))
 
         # Link station selection variables with combination variables
         for combo in station_combinations:
             # If combo is selected, corresponding stations must be selected
             for i in combo:
                 self.model.constraints.append(pk.constraint(self.model.x[i] >= self.model.combo_vars[combo]))
             # If all stations in combo are selected, combo must be selected
             self.model.constraints.append(
                 pk.constraint(sum(self.model.x[i] for i in combo) - self.max_stations * self.model.combo_vars[combo] <= self.max_stations - 1)
             )
 
         # Calculate total gap time for each combination using gap_times_condense
         total_gap_expr = 0
         for combo in station_combinations:
             # Collect all contacts for this combination
             combo_contacts = []
             for sat in self.satellites:
                 combo_gap = 0
                 for station_id in combo:
                     contacts = ba.find_location_accesses(sat, self.station_contacts[station_id]['station'], self.epc0, self.epc1)
                     combo_contacts.extend(contacts)
 
                 # Use gap_times_condense to calculate gaps
                 _, gaps_seconds = gap_times_condense(combo_contacts, self.epc0.to_datetime(), self.epc1.to_datetime())
                 combo_gap += sum(gaps_seconds)
 
 
             # Add this combination's contribution to total gap
             total_gap_expr += combo_gap * self.model.combo_vars[combo]
 
         # Set total gap constraint
         self.model.constraints.append(pk.constraint(self.model.total_gap == total_gap_expr))
 
         # Create and attach objective to model
         self.model.obj = pk.objective(self.model.total_gap, sense=pk.minimize)
 
         self.solve()
 
         # Print results
         selected_stations = []
         for i in self.model.x:
             if self.model.x[i].value > 0.5:
                 selected_stations.append(i)
 
         # Calculate and print actual gaps using gap_times_condense
         selected_contacts = []
         for station_id in selected_stations:
             selected_contacts.extend(self.station_contacts[station_id]['contacts'])
 
 
         all_gap_times, gaps_seconds = gap_times_condense(selected_contacts, self.epc0.to_datetime(), self.epc1.to_datetime())
         print("\nGaps between merged contacts:")
         for i, gap_time in enumerate(gaps_seconds):
             print(f"Gap {i+1}: {gap_time:.2f} seconds")
 
         print(f"\nTotal gap time: {sum(gaps_seconds):.2f} seconds")
         print([self.station_contacts[i]['name'] for i in selected_stations])
         return selected_stations, self.station_contacts, all_gap_times

    def max_contact_ilp(self):
        """
            max_contact_ilp() -> (list, dict, int)

        Select `max_stations` sites to maximize the number of usable contacts.

        NOTE: still written against the pre-1.0 brahe access API (`ba` is not imported); the
        contact-count objective is not currently supported.
        """
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
