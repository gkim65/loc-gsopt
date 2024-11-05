import numpy as np

from global_land_mask import globe

import random_land_points as rlp

def centroid(best, second):
    x = (best[0]+second[0])/2
    y = (best[0]+second[0])/2
    return (x,y)

# originally was reflection function, but can be used to move all points
def move_point(point_away, point_towards, factor):
    in_water = True

    points_wrong = []
    counter = 0
    while in_water:
        print(point_towards)
        print(point_away)
        # slope = (point_towards[0]-point_away[0])/(point_towards[1]-point_away[1])

        lon = point_towards[0]+(point_towards[0]-point_away[0])*factor
        lat = point_towards[1]+(point_towards[1]-point_away[1])*factor
        
        # if lon < -180:
        #     lon = -180
        # elif lon > 180:
        #     lon = 180
        
        # if lat < -90:
        #     lat = -90
        # elif lat > 90:
        #     lat = -60

        in_water = not(rlp.is_in(np.array([lon,lat]), continent = 'Antarctica')) # Check if in specified polygon? (duncan function?)
        if in_water: 
            factor = factor/2
            points_wrong.append((lon,lat))
        counter +=1 
        if counter > 10:
            print("IT DIDN'T WORK")
            new = rlp.random_points('Antarctica')[0] # Get a random point on land in Antartica
            return (new[0],new[1])

    return (lon,lat)


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


def nelder_mead(points, func, limit):
    
    plot_points = []
    plot_points.append(points)

    for i in range(limit):
        print("ITERATION: ",i)
        best, worst, second, best_val, worst_val, second_val = ranking(points, func)

        # first perform reflection
        bs_centroid = centroid(best, second)
        print("REFLECTION")
        temp = move_point(worst,bs_centroid,1)
        temp_val = func(temp)

        # if temp better than best value, perform expansion
        if temp_val < best_val:
            print("EXPANSION")
            temp_exp = move_point(bs_centroid,temp,1) 
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
            temp_cont = move_point(worst,bs_centroid,-0.5) 
            temp_cont_val = func(temp)
            if temp_cont_val > temp_val:
                worst = temp_cont
            else:
                worst = temp

        # not better than anything, shrink
        else: 
            print("SHRINK")
            worst = move_point(worst, best, -0.5)
            second = move_point(second, best, -0.5)

        points = [best, second, worst]
        plot_points.append(points)
    
    return plot_points
