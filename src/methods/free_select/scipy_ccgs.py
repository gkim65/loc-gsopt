import numpy as np
from scipy.optimize import minimize
from common.objective_functions import cost_func
import brahe as bh
from common.utils import compute_contact_times, xyz_to_latlon, latlon_to_xyz, contactExclusion

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon


def haversine_numba(lat1, lon1, lat2, lon2):
    """
        haversine_numba(lat1, lon1, lat2, lon2) -> float

    Great-circle distance on a 6371 km sphere (plain NumPy; the name predates dropping Numba).

      - `lat1`, `lon1`, `lat2`, `lon2` — the two points (deg)

    Returns the distance (km).
    """
    R = 6371.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi  = np.deg2rad(lat2 - lat1)
    dlamb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlamb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def farthest_four(points):
    """
        farthest_four(points) -> numpy.ndarray

    Pick 4 widely spaced points by greedy farthest-point selection from a random first point.

      - `points` — N x 2 array of `[lat, lon]` rows (deg)

    Returns the 4 chosen rows.
    """
    N = len(points)
    chosen = [np.random.randint(N)]          # 1st point random
    dists  = np.full(N, np.inf)

    for _ in range(3):                      # need total of 4
        # update min distance to current set
        for i in range(N):
            for j in chosen[-1:]:
                d = haversine_numba(points[i,0], points[i,1],
                                     points[j,0], points[j,1])
                if d < dists[i]:
                    dists[i] = d
        chosen.append(int(np.argmax(dists))) # pick farthest

    return points[chosen]

def simplex_select(n_samples=120):
    """
        simplex_select(n_samples=120) -> numpy.ndarray

    Initial Nelder-Mead simplex: `n_samples` points uniform on the sphere, reduced to 4 widely
    spaced ones by `farthest_four`.

      - `n_samples` — number of random points

    Returns a 4 x 2 array of `[lat, lon]` rows (deg).
    """
    lats = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_samples)))
    lons = np.random.uniform(-180, 180, n_samples)
    pts  = np.column_stack((lats, lons))
    return farthest_four(pts)




def nelder_mead_scipy_ccgs(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data, rng=None):
        """
            nelder_mead_scipy_ccgs(cfg, land_data, epc_start, epc_end, satellites, eval_counter, city_data, rng=None) -> (list, list)

        SCORE with a Nelder-Mead inner search on the unit sphere. The first pass places stations
        one at a time; each of the remaining `cfg.experiments.ccgs - 1` passes re-optimizes every
        station with the others fixed.

          - `cfg` — the Hydra config (`problem.gs_num`, `experiments.ccgs`, `experiments.ccgs_random`,
            `constraints`)
          - `land_data` — GeoDataFrame of country polygons
          - `epc_start`, `epc_end` — simulation window (`bh.Epoch`)
          - `satellites` — propagated `bh.SGPPropagator` objects
          - `eval_counter` — `EvalCounter`; `score_count` counts objective evaluations
          - `city_data` — GeoDataFrame of infrastructure zones
          - `rng` — NumPy generator that shuffles the refinement order when `ccgs_random`

        Returns the stations as `bh.PointLocation` objects and as `[lon, lat]` pairs (deg).

        NOTE: each search stops at 100 iterations; SciPy ignores the `xtol`/`ftol` options passed
        here and uses its default `xatol`/`fatol` of 1e-4.
        NOTE: the simplex rows are `[lat, lon]` but are converted with `latlon_to_xyz(p[1], p[0])`,
        i.e. with latitude and longitude swapped.
        NOTE: a refined location is kept when the raw contact time (before contact exclusion)
        improves; the penalties are not re-checked.
        NOTE: `constraints.extra_gs=True` fails with `IndexError` for `problem.gs_num` > 1.
        """
        # Setup args for minimize function
        if cfg.constraints.extra_gs:
                gs_list = [bh.PointLocation(cfg.constraints.extra_gs_lon,cfg.constraints.extra_gs_lat)]
                gs_list_plot = [[cfg.constraints.extra_gs_lon,cfg.constraints.extra_gs_lat]]
                gs_offset = 1

        else:
                gs_list = []
                gs_list_plot = []
                gs_offset = 0
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose

        if cfg.problem.gs_num == 1 and cfg.constraints.extra_gs:
                return gs_list, gs_list_plot

        for iterate in range(cfg.experiments.ccgs):

                
                # Determine ground station iteration order
                gs_indices = list(range(cfg.problem.gs_num))
                print(gs_indices)
                if iterate > 0 and cfg.experiments.ccgs_random:
                        # Shuffle in-place using the provided rng for reproducibility
                        rng.shuffle(gs_indices)

                # for every ground station
                for i in gs_indices:
                        if iterate == 0 and cfg.constraints.extra_gs:
                                break

                        # Unused by Nelder-Mead when initial_simplex is given
                        initial_guess = np.array([-0.41244896,  0.6765351 ,  0.61007058])

                        initial_simplex = simplex_select(n_samples=200)
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

                        # Contact time of the other stations, held fixed during this search
                        if iterate > 0:
                                # Get all ground stations except the i-th one
                                gs_list_others = [gs for idx, gs in enumerate(gs_list) if idx != i]
                                # try to minimize number of contacts to compute:
                                contacts_og, contacts_sec = compute_contact_times(satellites, gs_list_others ,epc_start, epc_end)
                                gs_contacts_og = contacts_sec

                        result = minimize(cost_func, 
                                        initial_guess, 
                                        args = (gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og,eval_counter,city_data, verbose, False), 
                                        method='Nelder-Mead',
                                        options={'disp': True,
                                                'maxiter': 100,
                                                'xtol': 1,     # NOTE: ignored by SciPy Nelder-Mead
                                                'ftol': 1,     # NOTE: ignored by SciPy Nelder-Mead
                                                'initial_simplex': np.array(simplex)})

                        
                        if cfg.debug.verbose:
                                print("GS FOUND, Location: "+str(result.x))

                        # conversion of unit circle coordinates back to lon,lat
                        if iterate == 0:
                                coord = xyz_to_latlon(result.x)
                                gs_list.append(bh.PointLocation(coord[1], coord[0]))
                                gs_list_plot.append([coord[1], coord[0]])

                                # try to minimize number of contacts to compute:
                                contacts_og, contacts_sec = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                                if i < cfg.problem.gs_num-1: # prevent double counting
                                        gs_contacts_og = contacts_sec
                        else:
                                coord = xyz_to_latlon(result.x)
                                gs_list_new = gs_list.copy()
                                gs_list_new[i] = bh.PointLocation(coord[1], coord[0])

                                # Keep the move only if the network's raw contact time improves
                                contacts_prev, contacts_sec_prev = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                                contacts_new, contacts_sec_new = compute_contact_times(satellites, gs_list_new ,epc_start, epc_end)

                                if np.sum(contacts_sec_prev) < np.sum(contacts_sec_new):
                                        print(gs_list)
                                        print(gs_list_plot)
                                        gs_list[i] = bh.PointLocation(coord[1], coord[0])
                                        gs_list_plot[i] = [coord[1], coord[0]]

                

                if cfg.debug.wandb:
                        contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
                        wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
                        wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
                        wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate
                        wandb.summary["eval_counter"+str(iterate)] = eval_counter.score_count

                               
                
        return gs_list, gs_list_plot 



def powell_scipy(cfg,land_data,epc_start,epc_end,satellites,eval_counter,city_data):
        """
            powell_scipy(cfg, land_data, epc_start, epc_end, satellites, eval_counter, city_data) -> (list, list)

        SCORE with a bounded Powell inner search in longitude and latitude, using the same
        place-then-refine passes as `nelder_mead_scipy_ccgs`.

          - `cfg` — the Hydra config (`problem.gs_num`, `experiments.ccgs`, `constraints`)
          - `land_data` — GeoDataFrame of country polygons
          - `epc_start`, `epc_end` — simulation window (`bh.Epoch`)
          - `satellites` — propagated `bh.SGPPropagator` objects
          - `eval_counter` — `EvalCounter`; `score_count` counts objective evaluations
          - `city_data` — GeoDataFrame of infrastructure zones

        Returns the stations as `bh.PointLocation` objects and as `[lon, lat]` pairs (deg).

        NOTE: start points come from Python's `random`, which `debug.randseed` does not seed, so
        runs are not reproducible from the seed.
        NOTE: the land penalty does not force a station onto land; a short search can end over
        the ocean.
        """
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
                
                        # Latitude: Uniform sampling between -90 and 90 degrees
                        lat = random.uniform(-90, 90)
                        
                        # Longitude: Uniform sampling between -180 and 180 degrees
                        lon = random.uniform(-180, 180)

                        initial_guess = np.array([lon,lat])

                        if cfg.debug.verbose:
                                print("STARTING TO PERFORM MINIMIZATION ON GS: "+str(i+1))

                        if iterate > 0:
                                gs_list_others = [gs for idx, gs in enumerate(gs_list) if idx != i]
                                contacts_og, contacts_sec = compute_contact_times(
                                        satellites, gs_list_others, epc_start, epc_end
                                )
                                gs_contacts_og = contacts_sec

                        result = minimize(
                                        cost_func,
                                        initial_guess,
                                        args=(gs_list, sat_list, epc_start, epc_end, land_geometries, cfg, i, gs_contacts_og,eval_counter,city_data, verbose, False), 
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

                        if iterate == 0:
                                gs_list.append(bh.PointLocation(result.x[0], result.x[1]))
                                gs_list_plot.append([result.x[0], result.x[1]])

                        else:
                                coord = result.x
                                gs_list_new = gs_list.copy()
                                gs_list_new[i] = bh.PointLocation(coord[0], coord[1])

                                contacts_prev, contacts_sec_prev = compute_contact_times(
                                satellites, gs_list, epc_start, epc_end
                                )
                                contacts_new, contacts_sec_new = compute_contact_times(
                                satellites, gs_list_new, epc_start, epc_end
                                )

                                if np.sum(contacts_sec_prev) < np.sum(contacts_sec_new):
                                        gs_list[i] = bh.PointLocation(coord[0], coord[1])
                                        gs_list_plot[i] = [coord[0], coord[1]]
                        # Contact time of the stations placed so far, for the next search
                        contacts_og, contacts_sec = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        gs_contacts_og = contacts_sec

                if cfg.debug.wandb:
                        contacts, _ = compute_contact_times(satellites, gs_list ,epc_start, epc_end)
                        _, contacts_exclusion_secs = contactExclusion(contacts,cfg)
                        wandb.summary["gs_list"+str(iterate)] = gs_list_plot 
                        wandb.summary["contact_num"+str(iterate)] = len(contacts_exclusion_secs) 
                        wandb.summary["seconds"+str(iterate)] = np.sum(contacts_exclusion_secs)
                        wandb.summary["data_downlink"+str(iterate)] = np.sum(contacts_exclusion_secs)*cfg.scenario.datarate
                        wandb.summary["eval_counter"+str(iterate)] = eval_counter.score_count
                
        return gs_list, gs_list_plot
