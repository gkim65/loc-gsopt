import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
from common.station_gen import return_bdm_gs
from common.utils import mp_compute_contact_times, xyz_to_latlon, latlon_to_xyz, contactExclusion

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon

def points_in_shape(gs_list, comb):
    for p in gs_list:
        triangle = Polygon(comb)
        point = Point(p)
        if triangle.contains(point) or triangle.touches(point):
            return True
    return False

def planar_quad_area(points):
    """
    Compute the area of a quadrilateral in 2D using the shoelace formula.
    `points` should be a 4x2 numpy array or list of 4 (x, y) tuples.
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    # Wrap around to first point
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.dot(x, y_next) - np.dot(y, x_next))
    return area

# TODO testing initial simplexes and seeing if mine's better
def simplex_rand():
        return ""

def simplex_select(gs_list,exclude, l=1):
               
        # try out 40 different starting points 
        lats = np.concatenate([np.random.uniform(-70, -50, 10),
                        np.random.uniform(50, 70, 10),
                        np.random.uniform(-50, 50, 10),
                        np.random.uniform(-50, 50, 10)])

        lons = np.concatenate([np.random.uniform(-160, 160, 10),
                        np.random.uniform(-160, 160, 10),
                        np.random.uniform(-160, -140, 10),
                        np.random.uniform(140, 160, 10)])


        # Stack latitudes and longitudes together
        all_points = np.column_stack([lats, lons])


        # Find the triangle with the largest area
        max_area = 0
        best_simplex = None

        # Generate all combinations of 3 points on the convex hull
        for comb in combinations(all_points, 4):
                if exclude:
                        if not points_in_shape(gs_list, comb):
                                # area = triangle_area(comb[0], comb[1], comb[2])
                                area = planar_quad_area(comb)
                                if area > max_area:
                                        max_area = area
                                        best_simplex = comb #np.array([comb[0], comb[1], comb[2]])
                else:
                        # area = triangle_area(comb[0], comb[1], comb[2])
                        area = planar_quad_area(comb)
                        if area > max_area:
                                max_area = area
                                best_simplex = comb #np.array([comb[0], comb[1], comb[2]])

        # TODO EXTENSION: possibly better simplex?
        # this lets me choose a gs as one of the points of the rectangle
        # for gs in gs_list:
        #         # Remove gs from gs_list for this   iteration
        #         other_gs = [pt for pt in gs_list if not np.allclose(pt, gs)]

        #         for pair in combinations(all_points, 2):
        #                 triangle = np.array([gs, pair[0], pair[1]])
        #                 print(points_in_shape(other_gs, triangle))
        #                 # Check that other gs points aren't inside the triangle
        #                 if not points_in_shape(other_gs, triangle):
        #                         area = triangle_area(triangle[0], triangle[1], triangle[2])
        #                         if area > max_area:
        #                                 max_area = area
        #                                 best_simplex = triangle
        return best_simplex



# TODO EXTENSION: add the cyclic coordinate descent part (May not include in this paper)
def nelder_mead_scipy_ccgs(cfg,land_data,epc_start,epc_end,satellites):

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose
        for iterate in range(cfg.experiments.ccgs):

                # for every ground station
                for i in range(cfg.problem.gs_num):
                

                        # Initial guess overridden by initial simplex
                        initial_guess = np.array([-0.41244896,  0.6765351 ,  0.61007058])

                        initial_simplex = simplex_select(gs_list_plot,cfg.experiments.simplexExclude)
                        if cfg.debug.wandb:
                                wandb.log({"initial simplex"+str(i+1): initial_simplex})
                                
                        # conversion of simplex to unit sphere
                        simplex = []
                        for p in initial_simplex:
                                simplex.append(latlon_to_xyz(p[1], p[0]))

                        if cfg.debug.verbose:
                                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))
                                print("lat-long simplex: ", initial_simplex)
                                print("unit circle converted simplex: ", simplex)

                        # # Perform the optimization using Nelder-Mead
                        result = minimize(cost_func, 
                                        initial_guess, 
                                        args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, False), 
                                        method='Nelder-Mead',
                                        options={'disp': True,
                                                'xtol': 1,     # x tolerance
                                                'ftol': 1,     # function tolerance
                                                'initial_simplex': np.array(simplex)})#,
                                                # 'maxiter': 3})

                        
                        if cfg.debug.verbose:
                                print("GS FOUND, Location: "+str(result.x))

                        # conversion of unit circle coordinates back to lon,lat
                        if iterate == 0:
                                coord = xyz_to_latlon(result.x)
                                gs_list.append(return_bdm_gs(coord[1], coord[0]))
                                gs_list_plot.append([coord[1], coord[0]])

                                # try to minimize number of contacts to compute:
                                contacts_og, contacts_sec = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                                gs_contacts_og = contacts_sec
                        else:
                                coord = xyz_to_latlon(result.x)
                                gs_list_new = gs_list.copy()
                                gs_list_new[i] = return_bdm_gs(coord[1], coord[0])

                                # Check if prev is better than current
                                contacts_prev, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                                _, contacts_exclusion_secs_prev = contactExclusion(contacts_prev,cfg)
                                contacts_new, _ = mp_compute_contact_times(satellites, gs_list_new ,epc_start, epc_end, False)
                                _, contacts_exclusion_secs_new = contactExclusion(contacts_new,cfg)

                                if np.sum(contacts_exclusion_secs_prev) < np.sum(contacts_exclusion_secs_new):
                                        print(gs_list)
                                        print(gs_list_plot)
                                        gs_list[i] = return_bdm_gs(coord[1], coord[0])
                                        gs_list_plot[i] = [coord[1], coord[0]]
                

                if cfg.debug.wandb:
                        contacts, _ = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
                        wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
                        wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
                        wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate

                               
                
        return gs_list, gs_list_plot 



def powell_scipy(cfg,land_data,epc_start,epc_end,satellites):

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose

        # for every ground station
        for i in range(cfg.problem.gs_num):
        
                # Latitude: Uniform sampling between -90 and 90 degrees
                lat = random.uniform(-90, 90)
                
                # Longitude: Uniform sampling between -180 and 180 degrees
                lon = random.uniform(-180, 180)

                # Initial guess overridden by initial simplex
                initial_guess = np.array([lon,lat])

                initial_simplex = simplex_select(gs_list_plot,cfg.experiments.simplexExclude)
                if cfg.debug.wandb:
                        wandb.log({"initial simplex"+str(i+1): initial_simplex})
                        
                # conversion of simplex to unit sphere
                simplex = []
                for p in initial_simplex:
                       simplex.append(latlon_to_xyz(p[1], p[0]))

                if cfg.debug.verbose:
                        print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))
                        print("lat-long simplex: ", initial_simplex)
                        print("unit circle converted simplex: ", simplex)

                result = minimize(
                                cost_func,
                                initial_guess,
                                args=(gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, False), 
                                method='Powell',
                                bounds=[(-180, 180), (-90, 90)],
                                options={
                                        'disp': True,
                                        'xtol': 1e-3,     # x tolerance
                                        'ftol': 1e-1,     # function tolerance
                                        'maxiter': 100
                                }
                                )
                
                if cfg.debug.verbose:
                        print("GS FOUND, Location: "+str(result.x))

                # conversion of unit circle coordinates back to lon,lat
                gs_list.append(return_bdm_gs(result.x[0], result.x[1]))
                gs_list_plot.append([result.x[0], result.x[1]])

                # try to minimize number of contacts to compute:
                contacts_og, contacts_sec = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, False)
                gs_contacts_og = contacts_sec
                
        return gs_list, gs_list_plot 


##### TODO EXTENSION: Powell / other minimize functions?
                # result = minimize(
                #                 cost_func,
                #                 initial_guess,
                #                 args=(gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, plot), 
                #                 method='Powell',
                #                 bounds=[(-180, 180), (-90, 90)],
                #                 options={
                #                         'disp': True,
                #                         'xtol': 1e-3,     # x tolerance
                #                         'ftol': 1e-6,     # function tolerance
                #                         'maxiter': 100
                #                 }
                #                 )