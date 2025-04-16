import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
from common.station_gen import return_bdm_gs
from common.utils import mp_compute_contact_times, xyz_to_latlon, latlon_to_xyz

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon

def points_in_triangle_shapely(gs_list, comb):
    for p in gs_list:
        triangle = Polygon(comb)
        point = Point(p)
        if triangle.contains(point) or triangle.touches(point):
            return True
    return False


# Step 2: Calculate the area of triangles formed by each combination of 3 points
def triangle_area(p1, p2, p3):
        # Shoelace formula to calculate area of triangle
        return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2

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

def simplex_rand():

def simplex_select(gs_list):

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
        best_triangle = None

        # Generate all combinations of 3 points on the convex hull
        for comb in combinations(all_points, 4):
                if not points_in_triangle_shapely(gs_list, comb):
                        # area = triangle_area(comb[0], comb[1], comb[2])
                        area = planar_quad_area(comb)
                        if area > max_area:
                                max_area = area
                                best_triangle = comb #np.array([comb[0], comb[1], comb[2]])
        
        # this lets me choose a gs as one of the points of the rectangle
        # for gs in gs_list:
        #         # Remove gs from gs_list for this   iteration
        #         other_gs = [pt for pt in gs_list if not np.allclose(pt, gs)]

        #         for pair in combinations(all_points, 2):
        #                 triangle = np.array([gs, pair[0], pair[1]])
        #                 print(points_in_triangle_shapely(other_gs, triangle))
        #                 # Check that other gs points aren't inside the triangle
        #                 if not points_in_triangle_shapely(other_gs, triangle):
        #                         area = triangle_area(triangle[0], triangle[1], triangle[2])
        #                         if area > max_area:
        #                                 max_area = area
        #                                 best_triangle = triangle

        print(best_triangle)

        return best_triangle



# TODO: add the cyclic coordinate descent part (May not include in this paper)
def nelder_mead_scipy(cfg,land_data,epc_start,epc_end,satellites):
        

        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        
        sat_list = satellites[0:cfg.problem.sat_num]
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose
        plot = cfg.debug.plot

        for i in range(cfg.problem.gs_num):
        
                # Initial guess for the coordinates (x, y) 
                # Overridden currently by initial simplex
                initial_guess = np.array([-30, -60])
                initial_guess = np.array([-0.41244896,  0.6765351 ,  0.61007058])

                # TODO: Find best initial simplex to start, may need to change based on continents

                initial_simplex = simplex_select(gs_list_plot)
                # initial_simplex =  np.array([[-30, -60],[-100, -60],[-30, -90]])
                # initial_simplex =  np.array([[-180, -90],[180, 0],[-180, 90]])
                
                # initial_simplex =
                initial_simplex_list = [ np.array([
                        [-60, -85],  # Point 1 (near the Caribbean)
                        [151.2093, -33.8688],  # Point 2 (Sydney, Australia)
                        [-153.67891140271288, 55.17076207063536]  # Point 3 (near Alaska)
                        ]),
                np.array([
                        [37.7749, -122.4194],  # Point 1 (San Francisco, USA)
                        [-33.8688, 151.2093],  # Point 2 (Sydney, Australia)
                        [55.7558, 37.6176]     # Point 3 (Moscow, Russia)
                ]),
                
                np.array([
                        [40.7128, -74.0060],   # Point 1 (New York City, USA)
                        [-22.9068, -43.1729],  # Point 2 (Rio de Janeiro, Brazil)
                        [34.0522, -118.2437]   # Point 3 (Los Angeles, USA)
                ]),

                np.array([
                        [51.5074, -0.1278],    # Point 1 (London, UK)
                        [-34.6037, -58.3816],  # Point 2 (Buenos Aires, Argentina)
                        [35.6895, 139.6917]    # Point 3 (Tokyo, Japan)
                ]),

                np.array([
                        [45.4215, -75.6972],   # Point 1 (Ottawa, Canada)
                        [39.9042, 116.4074],   # Point 2 (Beijing, China)
                        [-37.8136, 144.9631]   # Point 3 (Melbourne, Australia)
                ]),

                np.array([
                        [-1.286389, 36.817223],  # Point 1 (Nairobi, Kenya)
                        [40.730610, -73.935242], # Point 2 (New York City, USA)
                        [39.0742, 21.8243]       # Point 3 (Athens, Greece)
                ])
                ]

                print(initial_simplex_list[cfg.scenario.start_simplex])

                
                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))

                # array = np.array(initial_simplex)
                print(initial_simplex)
                simplex = []
                for p in initial_simplex:
                       simplex.append(latlon_to_xyz(p[1], p[0]))
                print(simplex)
                # # Perform the optimization using Nelder-Mead
                result = minimize(cost_func, 
                                initial_guess, 
                                args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og, verbose, plot), 
                                method='Nelder-Mead',
                                options={'disp': True,
                                        'xtol': 1,     # x tolerance
                                        'ftol': 1,     # function tolerance
                                         'initial_simplex': np.array(simplex)},) # initial_simplex_list[cfg.scenario.start_simplex]},
                                #bounds = ((-180,180),(-90,90)))
                # ,
                #                          'disp': True,
                #                         'fatol': 1, # TODO: Doesn't work
                #                         'xatol': 1e-2, # this might be too much! # TODO: Doesn't work
                #                         # 'maxiter': 35, # this might be too little!
                #                         }, # TODO: initial_simplex_list[cfg.scenario.start_simplex]},
                #                 ) #bounds = ((-180,180),(-90,90)))

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
                        
                print("GS FOUND, Location: "+str(result.x))
                coord = xyz_to_latlon(result.x)
                gs_list.append(return_bdm_gs(coord[1], coord[0]))
                gs_list_plot.append([coord[1], coord[0]])

                # try to minimize number of contacts to compute:
                contacts_og, contacts_sec = mp_compute_contact_times(satellites, gs_list ,epc_start, epc_end, plot)
                gs_contacts_og = contacts_sec#contacts_og
                
        return gs_list, gs_list_plot 


