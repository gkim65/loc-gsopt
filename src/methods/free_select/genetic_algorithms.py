import numpy as np
from scipy.optimize import differential_evolution
from common.objective_functions import cost_func_diffEvolution
import brahe as bh
from common.utils import compute_contact_times, xyz_to_latlon, latlon_to_xyz, contactExclusion

#  WandB
import wandb
import copy

# simplex selection
import random
from itertools import combinations
from shapely.geometry import Point, Polygon


def diffEvolution(cfg,land_data,epc_start,epc_end,satellites,eval_counter,CITY_KDTREE):
        """
            diffEvolution(cfg, land_data, epc_start, epc_end, satellites, eval_counter, CITY_KDTREE) -> (list, list)

        Place all `cfg.problem.gs_num` stations at once with SciPy differential evolution over the
        flattened `[lon_1, lat_1, ...]` vector, bounded to [-180, 180] x [-90, 90] deg.

          - `cfg` — the Hydra config; `cfg.DE` sets `popsize`, `F` (mutation), `CR` (recombination)
            and `strategy`
          - `land_data` — GeoDataFrame of country polygons
          - `epc_start`, `epc_end` — simulation window (`bh.Epoch`)
          - `satellites` — propagated `bh.SGPPropagator` objects
          - `eval_counter` — `EvalCounter`; `de_count` counts objective evaluations
          - `CITY_KDTREE` — GeoDataFrame of infrastructure zones (`data/city_buffers.shp`)

        Returns the stations as `bh.PointLocation` objects and as `[lon, lat]` pairs (deg).
        """
        # Setup args for minimize function
        gs_list = []
        gs_list_plot = []
        gs_contacts_og = []
        sat_list = satellites
        land_geometries = land_data['geometry']
        verbose = cfg.debug.verbose
        count = 1

        if cfg.debug.verbose:
            print("STARTING TO PERFORM MINIMIZATION ON ITERATION: "+str(count))

        result = differential_evolution(cost_func_diffEvolution, 
                        bounds = [(-180.0, 180.0), (-90.0, 90.0)] * cfg.problem.gs_num,
                        args = (sat_list, epc_start, epc_end, land_geometries, cfg, count, eval_counter,CITY_KDTREE, verbose, False), 
                        popsize= cfg.DE.popsize,
                        mutation= cfg.DE.F,
                        recombination=cfg.DE.CR,
                        strategy=cfg.DE.strategy,)

                        
        if cfg.debug.verbose:
                print("Final GS list: "+str(result.x))

        print(result)
        gs_list_plot =  [[lon, lat] for lon, lat in zip(result.x[::2], result.x[1::2])]
        gs_list = [bh.PointLocation(coord[0], coord[1]) for coord in gs_list_plot] 
        
        if cfg.debug.wandb:
                wandb.summary["EvalCount"] = eval_counter.de_count

                               
                               
                
        return gs_list, gs_list_plot 

