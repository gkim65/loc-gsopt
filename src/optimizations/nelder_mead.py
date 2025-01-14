import numpy as np

# from global_land_mask import globe

import random_land_points as rlp
from geopy.distance import geodesic

def centroid(best, second):
    x = (best[0]+second[0])/2
    y = (best[1]+second[1])/2
    return (x,y)

# originally was reflection function, but can be used to move all points
def move_point(point_away, point_towards, factor, continent_name):
    in_water = True

    points_wrong = []
    counter = 0
    while in_water:
        print(point_towards)
        print(point_away)
        slope = (point_towards[0]-point_away[0])/(point_towards[1]-point_away[1])

        lon = point_towards[0]+(point_towards[0]-point_away[0])*factor
        lat = point_towards[1]+(point_towards[1]-point_away[1])*factor
        
        if lon < -180:
            print("lon < -180")
            lon = -180
            lat = (lon - point_towards[0])/slope+point_towards[1]
            factor = abs((point_towards[0]-lon))/abs(point_towards[0]-point_away[0])
        elif lon > 180:
            print("lon > 180")
            lon = 180
            lat = (lon - point_towards[0])/slope+point_towards[1]
            factor = abs((point_towards[0]-lon))/abs(point_towards[0]-point_away[0])
        
        if lat < -90:
            print(" lat < -90")
            lat = -90
            lon = slope*(lat-point_towards[1])+point_towards[0]
            factor = abs((point_towards[0]-lon))/abs(point_towards[0]-point_away[0])
        elif lat > 90:
            print("lat > 90")
            lat = 90
            lon = slope*(lat-point_towards[1])+point_towards[0]
            factor = abs((point_towards[0]-lon))/abs(point_towards[0]-point_away[0])

        in_water = not(rlp.is_in(np.array([lon,lat]), continent = continent_name)) # Check if in specified polygon? (duncan function?)
        if in_water: 
            factor = factor/2
            points_wrong.append((lon,lat))
            print(points_wrong)
        counter +=1 
        if counter > 10:
            print("IT DIDN'T WORK")
            new = rlp.random_points(continent_name)[0] # Get a random point on land in Antartica
            print((new[0],new[1]))
            return (new[0],new[1])

    return (lon,lat)



# kind of convoluted, but this gives the ranking and values of best to worst points
# probably should find a more optimized way to do this
def ranking(points, func):

    val = []
    for point in points:
        val.append(func(point))

    ind = [0,1,2]
    best_ind = np.argmin(val)
    worst_ind = np.argmax(val)
    ind.remove(best_ind)
    ind.remove(worst_ind)
    second_ind = ind[0]
    best = points[best_ind]
    worst = points[worst_ind]
    second = points[second_ind]
    
    return best, worst, second, val[best_ind], val[worst_ind], val[second_ind]


# TODO HERE!
# Just get the centroid of three points and the distances between each point to the centroid
# https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi

# latitude to meters
# https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters

# Given you're looking for a simple formula, this is probably the simplest way to do it, assuming that the Earth is a sphere with a circumference of 40075 km.

# Length in km of 1° of latitude = always 111.32 km

# Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360
def converged(points, max_meters = 100):
    x = [x_val for x_val,_ in points]
    y = [y_val for _,y_val in points]
    l = len(x)
    centroid_coords = (sum(x)/l,sum(y)/l)
    distances = []
    for (x,y) in points:
        # lat_diff_meters = (centroid_coords[1]-y) * 111320
        # lon_diff_meters = (centroid_coords[0]-x) * 40075000 * np.cos(lat_diff_meters) / 360
        coord1= points[2]
        coord2= points[1]
        coord1 = [centroid_coords[1],centroid_coords[0]]
        coord2 = [y,x]
        dist = geodesic(coord1, coord2).km
        # dist = np.sqrt(lon_diff_meters**2 + lat_diff_meters**2) # conversions above
        distances.append(dist)
        if dist > max_meters:
            print("not converged!")
            print(distances)
            return False, centroid_coords, distances
    return True, centroid_coords, distances


def nelder_mead(points, func, limit, continent_name):
    
    plot_points = []
    debug_points = []
    plot_points.append(points)

    for i in range(limit):
        print("ITERATION: ",i)
        best, worst, second, best_val, worst_val, second_val = ranking(points, func)

        # first perform reflection
        bs_centroid = centroid(best, second)
        print(bs_centroid)
        print("REFLECTION")
        temp = move_point(worst,bs_centroid,1,continent_name)
        temp_val = func(temp)

        # if temp better than best value, perform expansion
        if temp_val < best_val:
            print("EXPANSION")
            temp_exp = move_point(bs_centroid,temp,1,continent_name) 
            temp_exp_val = func(temp)
            if temp_exp_val > temp_val:
                worst = temp_exp
            else:
                worst = temp

        # if temp better than only second value, just keep value
        elif temp_val < second_val:
            print("BETTER THAN SECOND")
            worst = temp

        # if temp better than only worst, contract
        elif temp_val < worst_val:
            print("CONTRACTION")
            temp_cont = move_point(worst,bs_centroid,-0.5,continent_name) 
            temp_cont_val = func(temp)
            if temp_cont_val > temp_val:
                worst = temp_cont
            else:
                worst = temp

        # not better than anything, shrink
        else: 
            print("SHRINK")
            worst = move_point(worst, best, -0.5,continent_name)
            second = move_point(second, best, -0.5,continent_name)

        points = [best, second, worst]
        plot_points.append(points)
        
        converge_bool, centroid_coords, distances = converged(points)
        debug_points.append([converge_bool, centroid_coords, distances])
        if converge_bool:
            print("CONVERGED!")
            print(centroid_coords)
            print(distances)
            return plot_points
    
    return plot_points
